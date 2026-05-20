#!/usr/bin/env python3
"""
analyze_merged_vs_nonmerged.py
==============================
Wersja offline — importuje funkcje bezpośrednio z backend/app.py
i modele z similarity_backend/app.py. Nie wymaga uruchomionych serwisów.

Zakłada strukturę katalogów:
  ./                                 ← ten skrypt + notebook
  ./backend/app.py                   ← główny Flask backend
  ./similarity_backend/app.py        ← similarity/reranker backend

Usage:
  python analyze_merged_vs_nonmerged.py --dataset gutenqa_all --data-root ../data
  python analyze_merged_vs_nonmerged.py --dataset gutenqa_all --data-root ../data \
      --score-tol 0.05 --reranker-threshold 2.0 --output results.json -v
"""

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_DATA_ROOT          = "../data"
DEFAULT_SCORE_TOL          = 0.02
DEFAULT_RERANKER_THRESHOLD = 1.0

ROOT = Path(__file__).parent.resolve()


# ── lazy module loaders ───────────────────────────────────────────────────────

_backend    = None
_similarity = None


def _load_module(name: str, path: Path) -> object:
    """
    Load a Python file as a module without triggering Flask's auto-run.
    The trick: exec_module requires the name in the spec to match, so we
    register the module in sys.modules under the spec name first, then
    rename __name__ AFTER exec so `if __name__ == "__main__"` is already
    past by the time the code runs — which means Flask never calls app.run().
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    # Register under the real spec name so the loader is happy
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    # Now override __name__ so callers can check it if needed
    mod.__name__ = f"_{name}_imported"
    return mod


def _load_backend(data_root: str):
    """
    Import backend/app.py injecting DATA_ROOT so path helpers work
    without Flask running.
    """
    global _backend
    if _backend is not None:
        return _backend

    backend_path = ROOT / "backend" / "app.py"
    if not backend_path.exists():
        raise FileNotFoundError(f"backend/app.py not found at {backend_path}")

    os.environ["DATA_ROOT"] = str(Path(data_root).resolve())
    _backend = _load_module("backend_app", backend_path)
    return _backend


def _load_similarity():
    """
    Import similarity_backend/app.py, loading ML models once.
    First call takes ~30s; subsequent calls return cached module instantly.
    """
    global _similarity
    if _similarity is not None:
        return _similarity

    sim_path = ROOT / "similarity_backend" / "app.py"
    if not sim_path.exists():
        raise FileNotFoundError(f"similarity_backend/app.py not found at {sim_path}")

    print("Loading ML models (BGE-M3 + BGE-reranker) — first call only, ~30s…")
    _similarity = _load_module("similarity_app", sim_path)
    print(f"  Models loaded on device: {_similarity.device}")
    return _similarity


# ── direct data accessors (replacing HTTP calls) ──────────────────────────────

def _get_pair_slugs(dataset_slug: str) -> tuple[str, str]:
    if dataset_slug.endswith("_merged"):
        return dataset_slug, dataset_slug[: -len("_merged")]
    return dataset_slug + "_merged", dataset_slug


def _fetch_queries_direct(data_root: str, slug: str) -> list[dict]:
    """Mirrors POST /api/datasets/:slug/queries logic from backend/app.py"""
    b = _load_backend(data_root)

    groups = b.group_experiments_by_dataset()
    exps   = groups.get(slug)
    if exps is None:
        raise ValueError(f"Dataset '{slug}' not found. Available: {sorted(groups.keys())}")

    chunk_counts = {exp: b.get_experiment_metadata(exp).get("chunk_count", 0) for exp in exps}
    extra_meta   = b.get_processed_query_meta(slug)
    queries_map: dict = {}

    for exp in exps:
        meta         = b.get_experiment_metadata(exp)
        chunker_name = meta.get("chunker_name", exp)
        total_chunks = chunk_counts.get(exp, 0)

        for item in b.get_retrieved(exp):
            qid     = item["id"]
            qid_str = str(qid)
            if qid not in queries_map:
                queries_map[qid] = {
                    "id":             qid,
                    "contents":       item.get("contents", ""),
                    "free_text_answer": (item.get("metadata") or {}).get("free_text_answer", ""),
                    "extra_meta":     extra_meta.get(qid_str, {}),
                    "chunkers":       {},
                }
            relevant_ids = item.get("relevant") or []
            queries_map[qid]["chunkers"][exp] = {
                "chunker_name":       chunker_name,
                "retrieved_relevant": item.get("retrieved_relevant", False),
                "retrieved":          item.get("retrieved", []),
                "retrieved_scores":   item.get("retrieved_scores", []),
                "relevant":           relevant_ids,
                "relevant_count":     len(relevant_ids),
                "chunk_count":        total_chunks,
                "relevant_pct":       round(len(relevant_ids) / total_chunks * 100, 4) if total_chunks else None,
            }

        for q in b.get_queries(exp):
            qid     = q["id"]
            qid_str = str(qid)
            if qid not in queries_map:
                queries_map[qid] = {
                    "id":             qid,
                    "contents":       q.get("contents", ""),
                    "free_text_answer": (q.get("metadata") or {}).get("free_text_answer", ""),
                    "extra_meta":     extra_meta.get(qid_str, {}),
                    "chunkers":       {},
                }
            if exp not in queries_map[qid]["chunkers"]:
                relevant_ids = q.get("relevant") or []
                queries_map[qid]["chunkers"][exp] = {
                    "chunker_name":       chunker_name,
                    "retrieved_relevant": False,
                    "retrieved":          [],
                    "retrieved_scores":   [],
                    "relevant":           relevant_ids,
                    "relevant_count":     len(relevant_ids),
                    "chunk_count":        total_chunks,
                    "relevant_pct":       round(len(relevant_ids) / total_chunks * 100, 4) if total_chunks else None,
                }

    return sorted(queries_map.values(), key=lambda q: q["id"])


def _fetch_relevant_chunk_texts_direct(data_root: str, exp: str, query_id: str) -> dict:
    """Mirrors GET /api/experiments/:exp/query/:id/relevant-chunk-texts"""
    b        = _load_backend(data_root)
    passages = b.get_passages(exp)

    item = next((r for r in b.get_retrieved(exp) if str(r["id"]) == str(query_id)), None)
    if item is None:
        item = next((q for q in b.get_queries(exp) if str(q["id"]) == str(query_id)), None)
    if item is None:
        raise ValueError(f"Query '{query_id}' not found in experiment '{exp}'")

    relevant_ids     = item.get("relevant") or []
    retrieved_ids    = item.get("retrieved") or []
    retrieved_scores = item.get("retrieved_scores") or []
    score_map        = {rid: retrieved_scores[i] if i < len(retrieved_scores) else None
                        for i, rid in enumerate(retrieved_ids)}

    return {
        "query_id":    query_id,
        "query_text":  item.get("contents", ""),
        "exp":         exp,
        "chunker_name": b.get_experiment_metadata(exp).get("chunker_name", exp),
        "chunks": [
            {
                "id":              cid,
                "contents":        passages.get(cid, {}).get("contents", ""),
                "retrieval_score": score_map.get(cid),
                "was_retrieved":   cid in score_map,
            }
            for cid in relevant_ids
        ],
    }


def _fetch_metrics_direct(data_root: str, slug: str) -> list[dict]:
    """Mirrors GET /api/datasets/:slug/metrics"""
    b      = _load_backend(data_root)
    groups = b.group_experiments_by_dataset()
    return [
        {
            "exp":          exp,
            "chunker_name": b.get_experiment_metadata(exp).get("chunker_name", exp),
            "metrics":      b.get_results(exp).get("metrics", {}),
        }
        for exp in groups.get(slug, [])
    ]


def _compute_similarity_direct(query_text: str, documents: list[str]) -> list[dict]:
    """
    Runs BGE-M3 + BGE-reranker directly in-process.
    Replaces POST /similarity.
    """
    if not documents:
        return []
    s = _load_similarity()

    query_emb = s.embed_model.encode(
        query_text, convert_to_tensor=True, normalize_embeddings=True, device=s.device
    )
    doc_embs = s.embed_model.encode(
        documents, convert_to_tensor=True, normalize_embeddings=True,
        device=s.device, batch_size=32,
    )
    scores_before = s.util.cos_sim(query_emb, doc_embs)[0]
    scores_after  = s.reranker.compute_score([[query_text, doc] for doc in documents])

    return [
        {"document": doc, "score_before": float(sb), "score_after": float(sa)}
        for doc, sb, sa in zip(documents, scores_before, scores_after)
    ]


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class ChunkScore:
    chunk_id:         str
    contents_preview: str
    score_retrieval:  float
    score_reranker:   float


@dataclass
class MatchedCase:
    query_id:                 str
    query_text:               str
    chunker_name:             str
    nonmerged_hit_chunks:     list[ChunkScore]
    merged_close_chunks:      list[ChunkScore]
    nonmerged_best_retrieval: float
    merged_best_retrieval:    float
    merged_best_reranker:     float


# ── helpers ───────────────────────────────────────────────────────────────────

def preview(text: str, n: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


def find_exp_for_chunker(queries_data: list[dict], chunker_name: str) -> Optional[str]:
    for q in queries_data:
        for exp, c in q.get("chunkers", {}).items():
            if c.get("chunker_name") == chunker_name:
                return exp
    return None


# ── main analysis ─────────────────────────────────────────────────────────────

def analyze(
    dataset_slug:       str,
    data_root:          str,
    score_tol:          float,
    reranker_threshold: float,
    verbose:            bool,
) -> list[MatchedCase]:

    merged_slug, nonmerged_slug = _get_pair_slugs(dataset_slug)

    print(f"Merged dataset    : {merged_slug}")
    print(f"Non-merged dataset: {nonmerged_slug}")
    print(f"Data root         : {Path(data_root).resolve()}")
    print()

    print("Loading queries for merged dataset…")
    merged_queries = _fetch_queries_direct(data_root, merged_slug)
    print(f"  {len(merged_queries)} queries")

    print("Loading queries for non-merged dataset…")
    nonmerged_queries = _fetch_queries_direct(data_root, nonmerged_slug)
    print(f"  {len(nonmerged_queries)} queries")

    nonmerged_map = {str(q["id"]): q for q in nonmerged_queries}

    print("Loading metrics…")
    merged_metrics = _fetch_metrics_direct(data_root, merged_slug)
    chunker_names  = [m["chunker_name"] for m in merged_metrics]
    print(f"  Chunkers: {chunker_names}")
    print()

    # Trigger model load once before the loop
    _load_similarity()
    print()

    matched_cases: list[MatchedCase] = []
    total = len(merged_queries)

    for chunker_name in chunker_names:
        print(f"── Chunker: {chunker_name} ─────────────────────────────")

        merged_exp    = find_exp_for_chunker(merged_queries,    chunker_name)
        nonmerged_exp = find_exp_for_chunker(nonmerged_queries, chunker_name)

        if not merged_exp or not nonmerged_exp:
            print(f"  [SKIP] exp not found in both datasets")
            continue

        if verbose:
            print(f"  merged exp   : {merged_exp}")
            print(f"  nonmerged exp: {nonmerged_exp}")

        cases_found = 0

        for i, merged_q in enumerate(merged_queries):
            qid = str(merged_q["id"])

            if not verbose:
                print(f"  [{i+1}/{total}] {qid}…", end="\r")

            merged_cd   = merged_q.get("chunkers", {}).get(merged_exp)
            nonmerged_q = nonmerged_map.get(qid)
            if not merged_cd or not nonmerged_q:
                continue

            nonmerged_cd = nonmerged_q.get("chunkers", {}).get(nonmerged_exp)
            if not nonmerged_cd:
                continue

            # Gate: non-merged hit, merged missed
            if not (nonmerged_cd.get("retrieved_relevant") and not merged_cd.get("retrieved_relevant")):
                continue

            query_text = merged_q.get("contents", "")
            if verbose:
                print(f"\n  Query {qid}: non-merged HIT, merged MISS → computing similarity…")

            try:
                merged_data    = _fetch_relevant_chunk_texts_direct(data_root, merged_exp,    qid)
                nonmerged_data = _fetch_relevant_chunk_texts_direct(data_root, nonmerged_exp, qid)
            except Exception as e:
                print(f"\n  [WARN] chunk texts unavailable for {qid}: {e}")
                continue

            m_chunks  = merged_data.get("chunks",    [])
            nm_chunks = nonmerged_data.get("chunks", [])
            if not m_chunks or not nm_chunks:
                continue

            try:
                m_sims  = _compute_similarity_direct(query_text, [c["contents"] for c in m_chunks])
                nm_sims = _compute_similarity_direct(query_text, [c["contents"] for c in nm_chunks])
            except Exception as e:
                print(f"\n  [WARN] similarity failed for {qid}: {e}")
                continue

            for c, s in zip(m_chunks,  m_sims):
                c["_ret"] = s["score_before"]; c["_rer"] = s["score_after"]
            for c, s in zip(nm_chunks, nm_sims):
                c["_ret"] = s["score_before"]; c["_rer"] = s["score_after"]

            # Best retrieval score among retrieved non-merged relevant chunks
            nm_hit_ret = [c["_ret"] for c in nm_chunks if c.get("was_retrieved")]
            if not nm_hit_ret:
                best = max((c["_ret"] for c in nm_chunks), default=None)
                nm_hit_ret = [best] if best is not None else []
            if not nm_hit_ret:
                continue

            close_merged = []
            nm_hit_chunks = []

            # zbierz trafione non-merged (z pełnymi score'ami)
            for c in nm_chunks:
                if c.get("was_retrieved"):
                    nm_hit_chunks.append(c)

            if not nm_hit_chunks:
                continue
            nm_ret = 0
            for m in m_chunks:
                for nm in nm_hit_chunks:
                    if (
                        abs(m["_ret"] - nm["_ret"]) <= score_tol
                        and m["_ret"] != nm["_ret"]
                        and abs(m["_rer"] - nm["_rer"]) > reranker_threshold
                    ):
                        nm_ret = nm["_ret"]
                        close_merged.append(
                            ChunkScore(
                                chunk_id=m["id"],
                                contents_preview=preview(m.get("contents", "")),
                                score_retrieval=m["_ret"],
                                score_reranker=m["_rer"],
                            )
                        )
                        break  # unikamy duplikatów dla tego samego merged chunk
            if not close_merged:
                continue

            nm_hit_chunks_scores = [
                ChunkScore(
                    chunk_id=c["id"],
                    contents_preview=preview(c.get("contents", "")),
                    score_retrieval=c["_ret"],
                    score_reranker=c["_rer"],
                )
                for c in nm_hit_chunks
            ]

            case = MatchedCase(
                query_id=qid,
                query_text=query_text,
                chunker_name=chunker_name,
                nonmerged_hit_chunks=nm_hit_chunks_scores,
                merged_close_chunks=close_merged,
                nonmerged_best_retrieval=nm_ret,
                merged_best_retrieval=max(c.score_retrieval for c in close_merged),
                merged_best_reranker=max(c.score_reranker   for c in close_merged),
            )
            matched_cases.append(case)
            cases_found += 1

            if verbose:
                _print_case(case, score_tol, reranker_threshold)

        print(f"\n  Found {cases_found} matching cases for {chunker_name}")
        print()

    return matched_cases


# ── pretty printing ───────────────────────────────────────────────────────────

def _print_case(case: MatchedCase, score_tol: float, reranker_threshold: float):
    print(f"\n  {'─'*70}")
    print(f"  QUERY  {case.query_id}: {preview(case.query_text, 100)}")
    print(f"  CHUNKER: {case.chunker_name}")
    print()
    print("  Non-merged retrieved relevant chunks:")
    for c in case.nonmerged_hit_chunks:
        print(f"    [{c.chunk_id}]  ret={c.score_retrieval:.4f}  rer={c.score_reranker:.4f}")
        print(f"      {c.contents_preview}")
    print()
    print(f"  Merged relevant chunks: ret±{score_tol} of non-merged AND rer>{reranker_threshold}:")
    for c in case.merged_close_chunks:
        print(f"    [{c.chunk_id}]  ret={c.score_retrieval:.4f}  rer={c.score_reranker:.4f}")
        print(f"      {c.contents_preview}")


def print_summary(cases: list[MatchedCase], score_tol: float, reranker_threshold: float):
    print()
    print("=" * 72)
    print(f"SUMMARY  —  {len(cases)} matching case(s)")
    print(f"  |retrieval_diff| ≤ {score_tol}  AND  |reranker_diff| > {reranker_threshold}")
    print("=" * 72)

    by_chunker: dict[str, list] = {}
    for c in cases:
        by_chunker.setdefault(c.chunker_name, []).append(c)

    for chunker, chunk_cases in by_chunker.items():
        print(f"\nChunker: {chunker}  ({len(chunk_cases)} case(s))")

        for case in chunk_cases:
            print(f"\n  Query {case.query_id}: {preview(case.query_text, 90)}")

            # pokaż wszystkie trafione non-merged
            print(f"  Non-merged HIT chunks:")
            for nm in case.nonmerged_hit_chunks:
                print(f"    [{nm.chunk_id}]  ret={nm.score_retrieval:.4f}  rer={nm.score_reranker:.4f}")
                print(f"      {nm.contents_preview}")

            print(f"\n  Merged conflicting chunks (embedding≈, reranker≠):")

            for m in case.merged_close_chunks:
                print(f"    [{m.chunk_id}]  ret={m.score_retrieval:.4f}  rer={m.score_reranker:.4f}")
                print(f"      {m.contents_preview}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline merged vs non-merged chunker analysis."
    )
    parser.add_argument("--dataset",   "-d", required=True,
                        help="Dataset slug (e.g. gutenqa_all or gutenqa_all_merged)")
    parser.add_argument("--data-root", "-r", default=DEFAULT_DATA_ROOT,
                        help=f"Root data directory with pirb/ and processed/ (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--score-tol", type=float, default=DEFAULT_SCORE_TOL,
                        help=f"Max retrieval score difference (default: {DEFAULT_SCORE_TOL})")
    parser.add_argument("--reranker-threshold", type=float, default=DEFAULT_RERANKER_THRESHOLD,
                        help=f"Min reranker score for merged chunks (default: {DEFAULT_RERANKER_THRESHOLD})")
    parser.add_argument("--output", "-o", default=None, help="Save results as JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    cases = analyze(
        dataset_slug=args.dataset,
        data_root=args.data_root,
        score_tol=args.score_tol,
        reranker_threshold=args.reranker_threshold,
        verbose=args.verbose,
    )
    print_summary(cases, args.score_tol, args.reranker_threshold)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in cases], f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
