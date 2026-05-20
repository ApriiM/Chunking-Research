"""
Chunker Qualitative Analysis Module
====================================
Compares retrieval results across chunkers for the same dataset and queries.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import torch


# ─────────────────────────────────────────────
# Scoring models (lazy loading)
# ─────────────────────────────────────────────

_EMBED_MODEL = None
_RERANKER = None
_DEVICE = None
_USE_FP16 = None

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _init_models():
    """Initialize models once on first use."""
    global _EMBED_MODEL, _RERANKER, _DEVICE, _USE_FP16
    
    if _EMBED_MODEL is not None:
        return
    
    from sentence_transformers import SentenceTransformer
    from FlagEmbedding import FlagReranker
    
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _USE_FP16 = _DEVICE == "cuda"
    
    print(f"🔧 Inicjalizacja modeli na: {_DEVICE}")
    
    _EMBED_MODEL = SentenceTransformer("BAAI/bge-m3", device=_DEVICE)
    if _USE_FP16:
        _EMBED_MODEL.half()
    _EMBED_MODEL.max_seq_length = 8192
    
    _RERANKER = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=_USE_FP16)
    if _DEVICE == "cuda":
        _RERANKER.model.to(_DEVICE)
    
    print("✅ Modele załadowane")


def compute_similarity(query: str, documents: list[str]) -> list[dict]:
    """
    Compute retrieval scores (cosine sim) and reranker scores.
    
    Returns list of dicts with keys: document, score_before, score_after
    """
    if not query or not isinstance(documents, list):
        raise ValueError("query musi być stringiem, documents listą")
    
    if len(documents) == 0:
        return []
    
    _init_models()
    
    from sentence_transformers import util
    
    # Retrieval score (before reranking)
    query_emb = _EMBED_MODEL.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=_DEVICE,
        show_progress_bar=False
    )
    
    doc_embs = _EMBED_MODEL.encode(
        documents,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=_DEVICE,
        batch_size=32,
        show_progress_bar=False
    )
    
    scores_before = util.cos_sim(query_emb, doc_embs)[0]
    
    # Reranker score (after)
    pairs = [[query, doc] for doc in documents]
    scores_after = _RERANKER.compute_score(pairs)
    
    # Build results
    results = []
    for doc, s_before, s_after in zip(documents, scores_before, scores_after):
        results.append({
            "document": doc,
            "score_before": float(s_before),
            "score_after": float(s_after)
        })
    
    return results


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class ExperimentMeta:
    exp_name: str
    dataset_slug: str
    chunker_name: str
    group_name: str
    chunk_count: int
    query_count: int


@dataclass
class QueryResult:
    query_id: str
    question: str
    free_text_answer: str
    relevant_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    retrieved_scores: list[float]
    retrieved_relevant: bool  # True if any retrieved chunk is relevant
    exp_name: str
    chunker_name: str
    dataset_slug: str
    # Computed scores (filled later)
    retrieval_scores: dict[str, float] = field(default_factory=dict)  # {chunk_id: score}
    reranker_scores: dict[str, float] = field(default_factory=dict)   # {chunk_id: score}


@dataclass
class ChunkInfo:
    chunk_id: str
    contents: str
    parent_id: str
    original_id: str
    exp_name: str
    chunker_name: str


@dataclass
class QueryGroup:
    """All chunker results for a single (dataset, query_id) pair."""
    dataset_slug: str
    query_id: str
    question: str
    free_text_answer: str
    results: list[QueryResult] = field(default_factory=list)

    @property
    def any_retrieved_relevant(self) -> bool:
        return any(r.retrieved_relevant for r in self.results)


# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_experiment_meta(pirb_data_dir: Path, exp_name: str) -> ExperimentMeta:
    meta_path = pirb_data_dir / exp_name / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        m = json.load(f)
    return ExperimentMeta(
        exp_name=exp_name,
        dataset_slug=m["dataset_slug"],
        chunker_name=m["chunker_name"],
        group_name=m.get("group_name", ""),
        chunk_count=m.get("chunk_count", 0),
        query_count=m.get("query_count", 0),
    )


def load_passages(pirb_data_dir: Path, exp_name: str) -> dict[str, ChunkInfo]:
    """Returns {chunk_id: ChunkInfo}"""
    meta = load_experiment_meta(pirb_data_dir, exp_name)
    passage_path = pirb_data_dir / exp_name / "passages" / "passages.jsonl"
    chunks: dict[str, ChunkInfo] = {}
    for rec in _read_jsonl(passage_path):
        metadata = rec.get("metadata", {})
        chunks[rec["id"]] = ChunkInfo(
            chunk_id=rec["id"],
            contents=rec["contents"],
            parent_id=metadata.get("parentId", ""),
            original_id=metadata.get("original_id", ""),
            exp_name=exp_name,
            chunker_name=meta.chunker_name,
        )
    return chunks


def load_retrieved_documents(
    retrieved_dir: Path,
    exp_name: str,
    chunker_name: str,
    dataset_slug: str,
) -> dict[str, QueryResult]:
    """Returns {query_id: QueryResult}"""
    path = retrieved_dir / f"{exp_name}.jsonl"
    results: dict[str, QueryResult] = {}
    for rec in _read_jsonl(path):
        results[rec["id"]] = QueryResult(
            query_id=rec["id"],
            question=rec["contents"],
            free_text_answer=rec.get("metadata", {}).get("free_text_answer", ""),
            relevant_chunk_ids=rec.get("relevant", []),
            retrieved_chunk_ids=rec.get("retrieved", []),
            retrieved_scores=rec.get("retrieved_scores", []),
            retrieved_relevant=rec.get("retrieved_relevant", False),
            exp_name=exp_name,
            chunker_name=chunker_name,
            dataset_slug=dataset_slug,
        )
    return results


def load_full_documents(processed_dir: Path, dataset_slug: str) -> dict[str, str]:
    """Returns {doc_id: full_text}"""
    path = processed_dir / dataset_slug / "documents" / "documents.jsonl"
    docs: dict[str, str] = {}
    for rec in _read_jsonl(path):
        docs[rec["id"]] = rec["contents"]
    return docs


# ─────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────

def discover_experiments(
    pirb_data_dir: Path,
    retrieved_dir: Path,
    target_datasets: Optional[list[str]] = None,
    exclude_chunkers: Optional[list[str]] = None,
) -> list[tuple[ExperimentMeta, dict[str, QueryResult]]]:
    """
    Scans pirb_data_dir for experiments, loads metadata and retrieved docs.
    Filters by dataset and excludes specified chunkers.
    """
    target_datasets = set(target_datasets) if target_datasets else None
    exclude_chunkers = set(exclude_chunkers) if exclude_chunkers else set()

    experiments = []
    for exp_dir in sorted(pirb_data_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        meta_path = exp_dir / "metadata.json"
        retrieved_path = retrieved_dir / f"{exp_dir.name}.jsonl"
        if not meta_path.exists() or not retrieved_path.exists():
            continue

        meta = load_experiment_meta(pirb_data_dir, exp_dir.name)

        if target_datasets and meta.dataset_slug not in target_datasets:
            continue
        if meta.chunker_name in exclude_chunkers:
            continue

        query_results = load_retrieved_documents(
            retrieved_dir, exp_dir.name, meta.chunker_name, meta.dataset_slug
        )
        experiments.append((meta, query_results))

    return experiments


# ─────────────────────────────────────────────
# Score computation for groups
# ─────────────────────────────────────────────

def compute_scores_for_groups(
    groups: list[QueryGroup],
    chunk_lookup: dict[str, dict[str, ChunkInfo]],
    batch_size: int = 8,
) -> None:
    """
    Compute retrieval and reranker scores for all relevant chunks in each group.
    Modifies QueryResult objects in-place, filling retrieval_scores and reranker_scores.
    
    Batches queries for efficiency.
    """
    print("⏳ Obliczanie scores (retrieval + reranker)…")
    
    total_queries = sum(len(g.results) for g in groups)
    processed = 0
    
    for group in groups:
        for qr in group.results:
            if not qr.relevant_chunk_ids:
                continue
            
            # Get chunk texts
            exp_chunks = chunk_lookup.get(qr.exp_name, {})
            chunk_texts = []
            chunk_ids = []
            
            for cid in qr.relevant_chunk_ids:
                chunk = exp_chunks.get(cid)
                if chunk:
                    chunk_texts.append(chunk.contents)
                    chunk_ids.append(cid)
            
            if not chunk_texts:
                continue
            
            # Compute scores
            results = compute_similarity(qr.question, chunk_texts)
            
            # Store in QueryResult
            for cid, res in zip(chunk_ids, results):
                qr.retrieval_scores[cid] = res["score_before"]
                qr.reranker_scores[cid] = res["score_after"]
            
            processed += 1
            if processed % 10 == 0:
                print(f"   Przetworzono {processed}/{total_queries} zapytań")
    
    print(f"✅ Obliczono scores dla {processed} zapytań")

def build_query_groups(
    experiments: list[tuple[ExperimentMeta, dict[str, QueryResult]]],
    merge_datasets: bool = False,   # 🔥 NOWE
) -> dict[tuple[str, str], QueryGroup]:
    """
    Groups QueryResults by:
    - (query_id) if merge_datasets=True
    - (dataset_slug, query_id) otherwise
    """
    groups: dict[tuple[str, str], QueryGroup] = {}

    for meta, query_results in experiments:
        for qid, qr in query_results.items():

            key = qid if merge_datasets else (meta.dataset_slug, qid)

            if key not in groups:
                groups[key] = QueryGroup(
                    dataset_slug="MULTI" if merge_datasets else meta.dataset_slug,
                    query_id=qid,
                    question=qr.question,
                    free_text_answer=qr.free_text_answer,
                )

            groups[key].results.append(qr)

    return groups


# ─────────────────────────────────────────────
# Chunk text matching helpers
# ─────────────────────────────────────────────

def _chunks_share_prefix(texts: list[str], min_prefix: int) -> bool:
    if min_prefix <= 0 or len(texts) < 2:
        return True
    prefix = texts[0][:min_prefix]
    return all(t[:min_prefix] == prefix for t in texts[1:])


def _chunks_share_suffix(texts: list[str], min_suffix: int) -> bool:
    if min_suffix <= 0 or len(texts) < 2:
        return True
    suffix = texts[0][-min_suffix:]
    return all(t[-min_suffix:] == suffix for t in texts[1:])


# ─────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────

def filter_groups(
    groups: dict[tuple[str, str], QueryGroup],
    only_when_retrieved_relevant: bool = True,
    min_common_prefix: int = 0,
    min_common_suffix: int = 0,
    chunk_lookup: Optional[dict[str, dict[str, ChunkInfo]]] = None,
) -> list[QueryGroup]:
    """
    Applies filters and returns matching QueryGroups.

    chunk_lookup: {exp_name: {chunk_id: ChunkInfo}} – required for prefix/suffix checks.
    """
    filtered = []
    for group in groups.values():
        if only_when_retrieved_relevant and not group.any_retrieved_relevant:
            continue

        if (min_common_prefix > 0 or min_common_suffix > 0) and chunk_lookup:
            # Collect all relevant chunk texts per group
            all_texts: list[str] = []
            for qr in group.results:
                exp_chunks = chunk_lookup.get(qr.exp_name, {})
                for cid in qr.relevant_chunk_ids:
                    if cid in exp_chunks:
                        all_texts.append(exp_chunks[cid].contents)

            if len(all_texts) < 2:
                filtered.append(group)
                continue

            if not _chunks_share_prefix(all_texts, min_common_prefix):
                continue
            if not _chunks_share_suffix(all_texts, min_common_suffix):
                continue

        filtered.append(group)

    return filtered


# ─────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RED    = "\033[31m"
_DIM    = "\033[2m"
_SEP    = "─" * 80


def _truncate(text: str, max_chars: int = 400) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + f"… [{len(text)-max_chars} more chars]"


def print_group(
    group: QueryGroup,
    chunk_lookup: dict[str, dict[str, ChunkInfo]],
    max_chunk_chars: int = 400,
    show_scores: bool = True,
) -> None:
    print(f"\n{_SEP}")
    print(f"{_BOLD}{_CYAN}📋 Pytanie [{group.dataset_slug}] | ID: {group.query_id}{_RESET}")
    print(f"{_BOLD}{group.question}{_RESET}")
    if group.free_text_answer:
        print(f"{_DIM}   Odpowiedź: {group.free_text_answer}{_RESET}")
    print()

    for qr in sorted(group.results, key=lambda r: r.chunker_name):
        exp_chunks = chunk_lookup.get(qr.exp_name, {})
        found_label = f"{_GREEN}✔ RETRIEVED_RELEVANT{_RESET}" if qr.retrieved_relevant else ""
        print(f"  {_BOLD}{_YELLOW}Chunker: {qr.chunker_name}{_RESET}  {found_label}")
        print(f"  {_DIM}Eksperyment: {qr.exp_name}{_RESET}")

        # Show relevant chunks (ground truth)
        texts = []
        chunk_data = []

        for cid in qr.relevant_chunk_ids:
            chunk = exp_chunks.get(cid)
            text = _truncate(chunk.contents, max_chunk_chars) if chunk else "[brak tekstu]"
            texts.append(text)
            chunk_data.append((cid, chunk, text))

        # 👉 jeśli wszystkie teksty identyczne → pomiń
        if len(set(texts)) == 1:
            print(f"  {_DIM}Wszystkie relewantne chunki mają identyczny tekst, pomijam szczegóły.{_RESET}")
            break

        print(f"  {_DIM}Relevantne chunki ({len(qr.relevant_chunk_ids)}):{_RESET}")

        for cid, chunk, text in chunk_data:
            retrieved_pos = None
            if cid in qr.retrieved_chunk_ids:
                retrieved_pos = qr.retrieved_chunk_ids.index(cid) + 1
            
            # Scores
            retr_score = qr.retrieval_scores.get(cid)
            rerank_score = qr.reranker_scores.get(cid)
            scores_str = ""
            if show_scores and retr_score is not None:
                scores_str = f" {_CYAN}retrieval={retr_score:.4f}{_RESET}"
            if show_scores and rerank_score is not None:
                scores_str += f" {_CYAN}reranker={rerank_score:.4f}{_RESET}"
            
            pos_label = f" {_GREEN}[zwrócony @{retrieved_pos}]{scores_str}{_RESET}" if retrieved_pos else f" {_RED}[NIE zwrócony]{scores_str}{_RESET}"
            print(f"    {_DIM}└─ {cid}{pos_label}{_RESET}")
            print(f"       {text}")

        # Show top-retrieved chunks (up to 3)
        # if qr.retrieved_chunk_ids:
        #     print(f"  {_DIM}Top zwrócone chunki (pierwsze 3):{_RESET}")
        #     for rank, (cid, score) in enumerate(
        #         zip(qr.retrieved_chunk_ids[:3], qr.retrieved_scores[:3]), start=1
        #     ):
        #         chunk = exp_chunks.get(cid)
        #         text = _truncate(chunk.contents, max_chunk_chars) if chunk else "[brak tekstu]"
        #         is_rel = cid in qr.relevant_chunk_ids
        #         rel_label = f" {_GREEN}✔ relevant{_RESET}" if is_rel else ""
        #         score_str = f"  score={score:.4f}" if show_scores else ""
        #         print(f"    {_DIM}└─ [{rank}] {cid}{score_str}{rel_label}{_RESET}")
        #         print(f"       {text}")
        print()

    print(_SEP)


def print_groups(
    groups: list[QueryGroup],
    chunk_lookup: dict[str, dict[str, ChunkInfo]],
    max_chunk_chars: int = 400,
    show_scores: bool = True,
    max_groups: Optional[int] = None,
) -> None:
    total = len(groups)
    shown = groups[:max_groups] if max_groups else groups
    print(f"\n{_BOLD}Znaleziono grup: {total}  |  Wyświetlam: {len(shown)}{_RESET}\n")
    for g in shown:
        print_group(g, chunk_lookup, max_chunk_chars, show_scores)


# ─────────────────────────────────────────────
# DataFrame builder
# ─────────────────────────────────────────────

def build_dataframe(
    groups: list[QueryGroup],
    chunk_lookup: dict[str, dict[str, ChunkInfo]],
) -> pd.DataFrame:
    """
    Returns a flat DataFrame with one row per (query, chunker, relevant_chunk).
    Includes retrieval_score and reranker_score columns.
    """
    rows = []
    for group in groups:
        for qr in group.results:
            exp_chunks = chunk_lookup.get(qr.exp_name, {})
            for cid in qr.relevant_chunk_ids:
                chunk = exp_chunks.get(cid)
                rank = None
                score = None
                if cid in qr.retrieved_chunk_ids:
                    rank = qr.retrieved_chunk_ids.index(cid) + 1
                    score = qr.retrieved_scores[rank - 1] if rank <= len(qr.retrieved_scores) else None
                
                # Get computed scores
                retrieval_score = qr.retrieval_scores.get(cid)
                reranker_score = qr.reranker_scores.get(cid)
                
                rows.append({
                    "dataset_slug":       group.dataset_slug,
                    "query_id":           group.query_id,
                    "question":           group.question,
                    "free_text_answer":   group.free_text_answer,
                    "chunker_name":       qr.chunker_name,
                    "exp_name":           qr.exp_name,
                    "chunk_id":           cid,
                    "chunk_text":         chunk.contents if chunk else "",
                    "parent_id":          chunk.parent_id if chunk else "",
                    "retrieved_rank":     rank,       # None if not retrieved
                    "retrieved_score":    score,
                    "retrieval_score":    retrieval_score,  # BGE-M3 cosine sim
                    "reranker_score":     reranker_score,   # BGE-reranker score
                    "is_retrieved":       rank is not None,
                    "group_retrieved_relevant": qr.retrieved_relevant,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Useful derived columns
    df["retrieved_rank"] = df["retrieved_rank"].astype("Int64")
    return df


# ─────────────────────────────────────────────
# High-level convenience function
# ─────────────────────────────────────────────

def run_analysis(
    pirb_data_dir: str | Path,
    retrieved_dir: str | Path,
    processed_dir: str | Path,
    target_datasets: Optional[list[str]] = None,
    exclude_chunkers: Optional[list[str]] = None,
    only_when_retrieved_relevant: bool = True,
    min_common_prefix: int = 0,
    min_common_suffix: int = 0,
    max_chunk_chars: int = 400,
    show_scores: bool = True,
    max_groups: Optional[int] = None,
    print_results: bool = True,
    enable_scoring: bool = True,  # New parameter
) -> tuple[list[QueryGroup], dict[str, dict[str, ChunkInfo]], pd.DataFrame]:
    """
    End-to-end pipeline.

    Parameters
    ----------
    enable_scoring : bool
        If True, compute retrieval and reranker scores using BGE models.
        Requires GPU for best performance. Set to False to skip (faster but no scores).

    Returns
    -------
    (filtered_groups, chunk_lookup, dataframe)
    """
    pirb_data_dir = Path(pirb_data_dir)
    retrieved_dir = Path(retrieved_dir)
    processed_dir = Path(processed_dir)

    print("⏳ Wczytywanie eksperymentów…")
    experiments = discover_experiments(pirb_data_dir, retrieved_dir, target_datasets, exclude_chunkers)
    print(f"   Znaleziono {len(experiments)} eksperymentów.")

    print("⏳ Wczytywanie pasaży…")
    chunk_lookup: dict[str, dict[str, ChunkInfo]] = {}
    for meta, _ in experiments:
        chunk_lookup[meta.exp_name] = load_passages(pirb_data_dir, meta.exp_name)

    print("⏳ Grupowanie zapytań…")
    merge_datasets = target_datasets is not None and len(target_datasets) > 1
    all_groups = build_query_groups(experiments, merge_datasets=merge_datasets)

    print("⏳ Filtrowanie grup…")
    filtered = filter_groups(
        all_groups,
        only_when_retrieved_relevant=only_when_retrieved_relevant,
        min_common_prefix=min_common_prefix,
        min_common_suffix=min_common_suffix,
        chunk_lookup=chunk_lookup,
    )
    print(f"   Grup po filtracji: {len(filtered)}")

    # Compute scores if enabled
    if enable_scoring and filtered:
        compute_scores_for_groups(filtered, chunk_lookup)

    if print_results:
        print_groups(filtered, chunk_lookup, max_chunk_chars, show_scores, max_groups)

    print("⏳ Budowanie DataFrame…")
    df = build_dataframe(filtered, chunk_lookup)
    print(f"   Wierszy w DataFrame: {len(df)}")

    return filtered, chunk_lookup, df