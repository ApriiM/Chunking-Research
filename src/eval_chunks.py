import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.data_loader.core.schemas import (
    load_document_records_jsonl,
    load_passage_records_jsonl,
    load_query_records_jsonl,
)


DEFAULT_MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _disable_torchvision_imports() -> None:
    try:
        import transformers.utils.import_utils as import_utils

        import_utils._torchvision_available = False
    except Exception:
        pass


def _load_model(model_name: str, model_weights: Optional[str]):
    _disable_torchvision_imports()
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise ImportError("transformers is required to load embedding models") from exc

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if not hasattr(model, "encode"):
        model._fallback_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    if model_weights and os.path.exists(model_weights):
        try:
            import torch

            state_dict = torch.load(model_weights, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available() and hasattr(model, "cuda"):
            model = model.cuda()
    except Exception:
        pass
    if hasattr(model, "eval"):
        model.eval()
    return model


def _resolve_paths(
    meta_path: str,
    documents_path: Optional[str],
    queries_path: Optional[str],
    passages_path: Optional[str],
) -> Tuple[str, str, str, Dict]:
    meta = _read_json(meta_path)
    documents_path = documents_path or meta.get("documents_path")
    passages_path = passages_path or meta.get("output_path")
    if not documents_path:
        raise ValueError("documents_path not found in metadata; pass --documents-path")
    if not passages_path:
        raise ValueError("output_path not found in metadata; pass --passages-path")
    if queries_path is None:
        base_dir = os.path.dirname(documents_path)
        queries_path = os.path.join(base_dir, "queries.jsonl")
    return documents_path, queries_path, passages_path, meta


def _encode_texts(model, texts: List[str], batch_size: int, show_progress: bool, mode: str):
    if mode == "queries" and hasattr(model, "encode_queries"):
        encode_fn = model.encode_queries
    elif mode == "corpus" and hasattr(model, "encode_corpus"):
        encode_fn = model.encode_corpus
    elif hasattr(model, "encode"):
        encode_fn = model.encode
    elif hasattr(model, "_fallback_tokenizer"):
        return _encode_with_transformers(model, model._fallback_tokenizer, texts, batch_size, show_progress)
    else:
        raise ValueError("Model does not expose encode/encode_queries/encode_corpus")

    kwargs = {}
    if batch_size:
        kwargs["batch_size"] = batch_size
    if show_progress is not None:
        kwargs["show_progress_bar"] = show_progress

    try:
        return encode_fn(texts, **kwargs)
    except TypeError:
        return encode_fn(texts)


def _encode_with_transformers(model, tokenizer, texts: List[str], batch_size: int, show_progress: bool):
    try:
        import torch
    except Exception as exc:
        raise ImportError("torch is required for fallback encoding") from exc

    if batch_size <= 0:
        batch_size = 32

    outputs = []
    for i in _progress_iter(range(0, len(texts), batch_size), show_progress, "Encoding"):
        batch = texts[i : i + batch_size]
        model_inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if model.device.type == "cuda":
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
        with torch.no_grad():
            model_outputs = model(**model_inputs)
        token_embeddings = model_outputs[0]
        attention_mask = model_inputs["attention_mask"].unsqueeze(-1)
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        outputs.append(pooled)

    return torch.cat(outputs, dim=0)


def _normalize_embeddings(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def _to_numpy(embs) -> np.ndarray:
    if hasattr(embs, "detach"):
        return embs.detach().cpu().numpy()
    return np.asarray(embs)


def _calculate_k_values(max_chunks: int) -> List[int]:
    k_values = [1, 3, 5, 10, 20]
    n = 2
    while 10**n < 100 * max_chunks:
        k_values.append(10**n)
        n += 1
    return k_values


def _ranked_docs(doc_scores: Dict[str, float], k: int) -> List[str]:
    return [doc_id for doc_id, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def _evaluate_local(
    relevant_docs: Dict[str, Dict[str, int]],
    doc_results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    ndcg = {f"ndcg@{k}": 0.0 for k in k_values}
    _map = {f"map@{k}": 0.0 for k in k_values}
    recall = {f"recall@{k}": 0.0 for k in k_values}
    precision = {f"precision@{k}": 0.0 for k in k_values}
    mrr = {f"mrr@{k}": 0.0 for k in k_values}

    query_ids = list(relevant_docs.keys())
    num_queries = len(query_ids)
    if num_queries == 0:
        return ndcg, _map, recall, precision, mrr

    for query_id in query_ids:
        rel_set = set(relevant_docs.get(query_id, {}).keys())
        doc_scores = doc_results.get(query_id, {})
        n_rel = len(rel_set)

        for k in k_values:
            ranked = _ranked_docs(doc_scores, k)
            hits = [1 if doc_id in rel_set else 0 for doc_id in ranked]
            num_hits = sum(hits)

            # Precision / Recall
            precision[f"precision@{k}"] += num_hits / k if k > 0 else 0.0
            recall[f"recall@{k}"] += num_hits / n_rel if n_rel > 0 else 0.0

            # MAP@k
            if n_rel > 0:
                denom = min(n_rel, k)
                ap = 0.0
                running_hits = 0
                for idx, hit in enumerate(hits):
                    if hit:
                        running_hits += 1
                        ap += running_hits / (idx + 1)
                _map[f"map@{k}"] += ap / denom if denom > 0 else 0.0

            # nDCG@k
            dcg = 0.0
            for idx, hit in enumerate(hits):
                if hit:
                    dcg += 1.0 / np.log2(idx + 2)
            if n_rel > 0:
                ideal = sum(1.0 / np.log2(i + 2) for i in range(min(n_rel, k)))
                ndcg[f"ndcg@{k}"] += (dcg / ideal) if ideal > 0 else 0.0

            # MRR@k
            rr = 0.0
            for idx, hit in enumerate(hits):
                if hit:
                    rr = 1.0 / (idx + 1)
                    break
            mrr[f"mrr@{k}"] += rr

    for metric in (ndcg, _map, recall, precision, mrr):
        for k in metric:
            metric[k] /= num_queries

    return ndcg, _map, recall, precision, mrr


def _progress_iter(items: Iterable, enabled: bool, desc: str):
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=desc)


def _get_doc_results(
    results: Dict[str, Dict[str, float]],
    passage_to_doc: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    doc_results: Dict[str, Dict[str, float]] = {}
    for query_id, chunk_scores in results.items():
        docs: Dict[str, float] = {}
        for chunk_id, score in chunk_scores.items():
            doc_id = passage_to_doc.get(chunk_id)
            if doc_id is None:
                continue
            if (doc_id not in docs) or (score > docs[doc_id]):
                docs[doc_id] = float(score)
        doc_results[query_id] = docs
    return doc_results


def _build_results(
    query_ids: List[str],
    query_embeddings: np.ndarray,
    passage_embeddings: np.ndarray,
    passage_ids: List[str],
    top_k: int,
    show_progress: bool,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for idx, query_id in enumerate(
        _progress_iter(query_ids, show_progress, desc="Scoring queries")
    ):
        scores = passage_embeddings @ query_embeddings[idx]
        if top_k >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        results[query_id] = {
            passage_ids[i]: float(scores[i]) for i in top_idx
        }
    return results


def evaluate_chunks(
    passages_meta_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    model_weights: Optional[str] = None,
    documents_path: Optional[str] = None,
    queries_path: Optional[str] = None,
    passages_path: Optional[str] = None,
    output_path: Optional[str] = None,
    batch_size: int = 32,
    max_queries: Optional[int] = None,
    max_passages: Optional[int] = None,
    show_progress: bool = False,
    normalize: bool = True,
) -> Dict:
    documents_path, queries_path, passages_path, meta = _resolve_paths(
        passages_meta_path, documents_path, queries_path, passages_path
    )

    documents = load_document_records_jsonl(documents_path)
    queries = load_query_records_jsonl(queries_path)
    passages = load_passage_records_jsonl(passages_path)

    doc_ids = {doc.doc_id for doc in documents}
    filtered_passages = [p for p in passages if p.parent_id in doc_ids]
    if max_passages is not None:
        filtered_passages = filtered_passages[: max_passages]
    if not filtered_passages:
        raise ValueError("No passages matched documents for evaluation")

    query_ids: List[str] = []
    query_texts: List[str] = []
    relevant_docs: Dict[str, Dict[str, int]] = {}
    for query in queries:
        rel_docs = {doc_id: 1 for doc_id in query.relevant if doc_id in doc_ids}
        if not rel_docs:
            continue
        query_ids.append(query.query_id)
        query_texts.append(query.contents)
        relevant_docs[query.query_id] = rel_docs
        if max_queries is not None and len(query_ids) >= max_queries:
            break

    if not query_ids:
        raise ValueError("No queries with relevant docs were found for evaluation")

    model = _load_model(model_name, model_weights)

    query_embs = _encode_texts(
        model, query_texts, batch_size=batch_size, show_progress=show_progress, mode="queries"
    )
    passage_texts = [p.contents for p in filtered_passages]
    passage_embs = _encode_texts(
        model, passage_texts, batch_size=batch_size, show_progress=show_progress, mode="corpus"
    )

    query_embs = _to_numpy(query_embs)
    passage_embs = _to_numpy(passage_embs)
    if query_embs.ndim == 1:
        query_embs = query_embs[None, :]
    if passage_embs.ndim == 1:
        passage_embs = passage_embs[None, :]

    if normalize:
        query_embs = _normalize_embeddings(query_embs)
        passage_embs = _normalize_embeddings(passage_embs)

    doc_chunk_counts = Counter(p.parent_id for p in filtered_passages)
    max_chunks = max(doc_chunk_counts.values()) if doc_chunk_counts else 1
    k_values = _calculate_k_values(max_chunks)
    top_k = max(k_values)

    passage_ids = [p.passage_id for p in filtered_passages]
    passage_to_doc = {p.passage_id: p.parent_id for p in filtered_passages}
    results = _build_results(
        query_ids=query_ids,
        query_embeddings=query_embs,
        passage_embeddings=passage_embs,
        passage_ids=passage_ids,
        top_k=top_k,
        show_progress=show_progress,
    )
    doc_results = _get_doc_results(results, passage_to_doc)

    max_k = int(max(k_values) / max_chunks)
    eval_k = [k for k in k_values if k <= max_k]
    metrics_impl = "local"
    try:
        from mteb.evaluation.evaluators import RetrievalEvaluator

        metrics_impl = "mteb"
        ndcg, _map, recall, precision, _ = RetrievalEvaluator.evaluate(
            relevant_docs,
            doc_results,
            eval_k,
            ignore_identical_ids=True,
        )
        mrr, _ = RetrievalEvaluator.evaluate_custom(
            relevant_docs,
            doc_results,
            eval_k,
            "mrr",
        )
    except Exception:
        ndcg, _map, recall, precision, mrr = _evaluate_local(
            relevant_docs, doc_results, eval_k
        )

    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
        **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
    }
    if "ndcg_at_10" in scores:
        scores["main_score"] = scores["ndcg_at_10"]

    payload = {
        "meta_path": passages_meta_path,
        "documents_path": documents_path,
        "queries_path": queries_path,
        "passages_path": passages_path,
        "model_name": model_name,
        "model_weights": model_weights,
        "batch_size": batch_size,
        "normalize": normalize,
        "document_count": len(documents),
        "query_count": len(query_ids),
        "passage_count": len(filtered_passages),
        "raw_query_count": len(queries),
        "raw_passage_count": len(passages),
        "max_passages_per_doc": max_chunks,
        "k_values": k_values,
        "eval_k": eval_k,
        "metrics_impl": metrics_impl,
        "scores": scores,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chunker_name": meta.get("chunker_name"),
        "chunker_params": meta.get("chunker_params"),
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pre-chunked passages with a late-chunking-compatible evaluator."
    )
    parser.add_argument(
        "--passages-meta",
        required=True,
        help="Path to passages_*.meta.json produced by prepare_passages",
    )
    parser.add_argument("--documents-path", help="Override documents.jsonl path")
    parser.add_argument("--queries-path", help="Override queries.jsonl path")
    parser.add_argument("--passages-path", help="Override passages.jsonl path")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Embedding model name or local path",
    )
    parser.add_argument("--model-weights", default=None, help="Optional finetuned weights")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Embedding batch size"
    )
    parser.add_argument("--max-queries", type=int, default=None, help="Limit queries")
    parser.add_argument("--max-passages", type=int, default=None, help="Limit passages")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show embedding progress bars",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize embeddings before scoring",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional JSON path for metrics output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_chunks(
        passages_meta_path=args.passages_meta,
        model_name=args.model_name,
        model_weights=args.model_weights,
        documents_path=args.documents_path,
        queries_path=args.queries_path,
        passages_path=args.passages_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_queries=args.max_queries,
        max_passages=args.max_passages,
        show_progress=args.show_progress,
        normalize=args.normalize,
    )
    print(json.dumps(payload.get("scores", {}), ensure_ascii=False, indent=2))
    if args.output_path:
        print(f"Wrote metrics to {args.output_path}")


if __name__ == "__main__":
    main()
