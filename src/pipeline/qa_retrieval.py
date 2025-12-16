import json
import os
from typing import Any, Dict, List, Tuple

import yaml

from src.chunking import get_chunker
from src.chunking.base import Chunk
from src.data_loader import (
    QASample,
    get_dataset_loader,
    load_samples_jsonl,
    save_samples_jsonl,
)
from src.evaluation import get_evaluations
from src.evaluation.retrieval import evaluate_retrieval
from src.evaluation.retrieval_eval import retrieval_at_k
from src.io_utils import write_manifest
from src.retrieval.registry import get_retriever
from src.schemas import (
    ChunkRecord,
    RetrievalResultRecord,
    validate_chunks,
    validate_retrieval_results,
    validate_samples,
)


def _load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_samples(cfg: Dict[str, Any]) -> Tuple[str, list[QASample]]:
    dataset_cfg = cfg.get("dataset", {})
    preprocessed_path = dataset_cfg.get("preprocessed_path")
    use_preprocessed = dataset_cfg.get("use_preprocessed", True)
    save_preprocessed = dataset_cfg.get("save_preprocessed", True)

    if use_preprocessed and preprocessed_path and os.path.exists(preprocessed_path):
        samples = load_samples_jsonl(preprocessed_path)
        validate_samples(samples)
        return "preprocessed_cache", samples

    dataset_name = dataset_cfg.get("name", "poquad")
    loader = get_dataset_loader(dataset_name)
    load_kwargs = dict(dataset_cfg.get("params", {}))
    samples = loader(**load_kwargs)
    validate_samples(samples)

    if save_preprocessed and preprocessed_path:
        save_samples_jsonl(samples, preprocessed_path)

    return dataset_name, samples


def _load_chunks_jsonl(path: str) -> List[Chunk]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    chunks: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunks.append(
                Chunk(
                    text=rec.get("text", ""),
                    chunk_id=rec.get("chunk_id"),
                    metadata=rec.get("metadata", {}),
                )
            )
    return chunks


def _build_chunker(cfg: Dict[str, Any]):
    chunker_cfg = cfg.get("chunker") or {}
    name = chunker_cfg.get("name")
    params = chunker_cfg.get("params", {})
    if not name:
        raise ValueError("Config must include chunker.name")
    return get_chunker(name, params)


def _build_retriever(cfg: Dict[str, Any]):
    retrieval_cfg = cfg.get("retrieval") or {}
    method = retrieval_cfg.get("method", "tfidf")
    retriever = get_retriever(method)
    return retriever, int(retrieval_cfg.get("top_k", 5))


def _make_chunk_records(chunks: List[Chunk]) -> List[dict]:
    return [ChunkRecord.from_chunk(c).__dict__ for c in chunks]


def _maybe_save_jsonl(records: list[dict], path: str, overwrite: bool = False) -> None:
    if not path or records is None:
        return
    if os.path.exists(path) and not overwrite:
        base, ext = os.path.splitext(path)
        path = f"{base}_cached{ext}"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_qa_retrieval(config_path: str) -> Dict[str, Any]:
    cfg = _load_config(config_path)
    _, samples = _resolve_samples(cfg)

    chunker = _build_chunker(cfg)
    retriever, top_k = _build_retriever(cfg)

    chunker_cfg = cfg.get("chunker", {})
    precomputed_chunks_path = chunker_cfg.get("precomputed_chunks_path")
    chunks: List[Chunk] = []
    chunk_records: List[dict] = []

    if precomputed_chunks_path and os.path.exists(precomputed_chunks_path):
        chunks = _load_chunks_jsonl(precomputed_chunks_path)
        validate_chunks(chunks)
        chunk_records = _make_chunk_records(chunks)

        metrics, results = retrieval_at_k(
            samples,
            chunks,
            retriever=retriever,
            top_k=top_k,
            relevance=cfg.get("retrieval", {}).get("relevance", "substring"),
        )
    else:
        metrics, results, chunk_records = evaluate_retrieval(
            samples,
            chunker,
            retriever=retriever,
            top_k=top_k,
            relevance=cfg.get("retrieval", {}).get("relevance", "substring"),
            return_chunks=True,
        )
        validate_chunks([ChunkRecord(**cr) for cr in (chunk_records or [])])
        # rebuild chunk objects for downstream evals
        chunks = [
            Chunk(text=cr.get("text", ""), chunk_id=cr.get("chunk_id"), metadata=cr.get("metadata", {}))
            for cr in (chunk_records or [])
        ]

    validate_retrieval_results(results)

    # Run additional evaluations configured under evaluations: []
    eval_cfg = cfg.get("evaluations", [])
    evals = get_evaluations(eval_cfg)
    extra_metrics: Dict[str, float] = {}
    for name, fn in evals:
        if name == "retrieval_at_k":
            m, _ = fn(samples, chunks, top_k=top_k)
            extra_metrics.update({f"{name}.{k}": v for k, v in m.items()})
        elif name == "chunk_stats":
            m = fn(chunks)
            extra_metrics.update({f"{name}.{k}": v for k, v in m.items()})
        else:
            m = fn(samples, chunks)
            extra_metrics.update({f"{name}.{k}": v for k, v in m.items()})

    output_cfg = cfg.get("output", {})
    overwrite = bool(output_cfg.get("overwrite", False))
    _maybe_save_jsonl(
        [RetrievalResultRecord.from_result(r).__dict__ for r in results],
        output_cfg.get("results_path"),
        overwrite,
    )
    _maybe_save_jsonl(
        [ChunkRecord(**cr).__dict__ for cr in (chunk_records or [])],
        output_cfg.get("chunks_path"),
        overwrite,
    )

    manifest_path = output_cfg.get("manifest_path", "results/run_manifest.json")
    aggregate_metrics = {**metrics, **extra_metrics}
    write_manifest(
        manifest_path,
        {
            "config_path": config_path,
            "metrics": aggregate_metrics,
            "dataset": cfg.get("dataset", {}),
            "chunker": cfg.get("chunker", {}),
            "retrieval": cfg.get("retrieval", {}),
            "evaluations": eval_cfg,
            "output": {
                "results_path": output_cfg.get("results_path"),
                "chunks_path": output_cfg.get("chunks_path"),
                "manifest_path": manifest_path,
            },
        },
        overwrite=True,
    )

    return aggregate_metrics
