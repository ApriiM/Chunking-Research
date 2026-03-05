import argparse
import bisect
import copy
import json
import os
from typing import Any, Dict, List, Tuple


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_jsonl(rows: List[Dict[str, Any]], path: str, overwrite: bool) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _score_overlap(
    gold_start: int,
    gold_end: int,
    chunk_start: int,
    chunk_end: int,
    mode: str,
) -> float:
    overlap = _overlap_len(gold_start, gold_end, chunk_start, chunk_end)
    if overlap <= 0:
        return 0.0

    gold_len = max(1, gold_end - gold_start)
    if mode == "gold":
        # User-friendly default: fraction of the gold span covered by this chunk.
        return overlap / gold_len

    chunk_len = max(1, chunk_end - chunk_start)
    union = gold_len + chunk_len - overlap
    return overlap / max(1, union)


def _index_chunks(
    passages: List[Dict[str, Any]],
    chunk_start_key: str,
    chunk_end_key: str,
) -> Dict[str, Dict[str, Any]]:
    by_parent: Dict[str, List[Tuple[str, int, int]]] = {}
    skipped = 0

    for p in passages:
        parent_id = p.get("parentId")
        chunk_id = p.get("id")
        meta = p.get("metadata") or {}
        start = _safe_int(meta.get(chunk_start_key))
        end = _safe_int(meta.get(chunk_end_key))

        if (
            not isinstance(parent_id, str)
            or not isinstance(chunk_id, str)
            or start is None
            or end is None
            or end <= start
        ):
            skipped += 1
            continue

        by_parent.setdefault(parent_id, []).append((chunk_id, start, end))

    indexed: Dict[str, Dict[str, Any]] = {}
    for parent_id, chunks in by_parent.items():
        chunks.sort(key=lambda x: (x[1], x[2], x[0]))
        starts = [start for _, start, _ in chunks]
        max_chunk_len = max((end - start) for _, start, end in chunks)
        indexed[parent_id] = {
            "chunks": chunks,
            "starts": starts,
            "max_chunk_len": max_chunk_len,
        }

    print(
        f"[Remap] Indexed {sum(len(v) for v in by_parent.values())} chunks across {len(by_parent)} parent docs."
    )
    if skipped:
        print(f"[Remap] Skipped {skipped} chunks missing valid offsets/ids.")
    return indexed


def remap_queries_to_chunks(
    queries: List[Dict[str, Any]],
    chunks_by_parent: Dict[str, Dict[str, Any]],
    gold_start_key: str,
    gold_end_key: str,
    threshold: float,
    score_mode: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    missing_spans = 0
    no_candidates = 0
    no_matches = 0
    matched_total = 0

    for q in queries:
        q_new = copy.deepcopy(q)
        q_meta = q_new.get("metadata") or {}
        gold_start = _safe_int(q_meta.get(gold_start_key))
        gold_end = _safe_int(q_meta.get(gold_end_key))

        if gold_start is None or gold_end is None or gold_end <= gold_start:
            missing_spans += 1
            out.append(q_new)
            continue

        candidate_parents = [doc_id for doc_id in (q_new.get("relevant") or []) if isinstance(doc_id, str)]
        parent_indexes: List[Dict[str, Any]] = []
        for parent_id in candidate_parents:
            parent_index = chunks_by_parent.get(parent_id)
            if parent_index is not None:
                parent_indexes.append(parent_index)

        if not parent_indexes:
            no_candidates += 1
            q_new["relevant"] = []
            out.append(q_new)
            continue

        matched: List[Tuple[str, float, int]] = []
        for parent_index in parent_indexes:
            chunks: List[Tuple[str, int, int]] = parent_index["chunks"]
            starts: List[int] = parent_index["starts"]
            max_chunk_len: int = parent_index["max_chunk_len"]

            # Restrict to chunks that can possibly overlap [gold_start, gold_end).
            lo = bisect.bisect_left(starts, gold_start - max_chunk_len + 1)
            hi = bisect.bisect_left(starts, gold_end)

            for chunk_id, chunk_start, chunk_end in chunks[lo:hi]:
                score = _score_overlap(gold_start, gold_end, chunk_start, chunk_end, mode=score_mode)
                if score >= threshold:
                    matched.append((chunk_id, score, chunk_start))

        matched.sort(key=lambda x: (x[2], -x[1], x[0]))
        q_new["relevant"] = [chunk_id for chunk_id, _, _ in matched]

        if matched:
            matched_total += len(matched)
        else:
            no_matches += 1

        out.append(q_new)

    print(f"[Remap] Processed {len(queries)} queries.")
    print(f"[Remap] Queries with missing/invalid gold span: {missing_spans}")
    print(f"[Remap] Queries with no candidate chunks: {no_candidates}")
    print(f"[Remap] Queries with zero matched chunks at threshold={threshold}: {no_matches}")
    print(f"[Remap] Total matched relevant chunk links: {matched_total}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite query relevant doc IDs to relevant chunk IDs based on overlap with gold span."
    )
    parser.add_argument("--queries-path", required=True, help="Input queries JSONL path")
    parser.add_argument("--passages-path", required=True, help="Input chunked passages JSONL path")
    parser.add_argument("--output-queries-path", required=True, help="Output remapped queries JSONL path")
    parser.add_argument(
        "--gold-start-key",
        default="document_start",
        help="Query metadata key holding gold span start offset",
    )
    parser.add_argument(
        "--gold-end-key",
        default="document_end",
        help="Query metadata key holding gold span end offset",
    )
    parser.add_argument(
        "--chunk-start-key",
        default="start_char",
        help="Chunk metadata key holding chunk span start offset",
    )
    parser.add_argument(
        "--chunk-end-key",
        default="end_char",
        help="Chunk metadata key holding chunk span end offset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Match threshold. With --score-mode gold, this is required fraction of gold span covered by chunk.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["gold", "iou"],
        default="gold",
        help="gold: overlap/gold_span_len (default). iou: overlap/union.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting output file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("--threshold must be in [0, 1]")

    queries = _load_jsonl(args.queries_path)
    passages = _load_jsonl(args.passages_path)

    chunks_by_parent = _index_chunks(
        passages=passages,
        chunk_start_key=args.chunk_start_key,
        chunk_end_key=args.chunk_end_key,
    )
    remapped = remap_queries_to_chunks(
        queries=queries,
        chunks_by_parent=chunks_by_parent,
        gold_start_key=args.gold_start_key,
        gold_end_key=args.gold_end_key,
        threshold=args.threshold,
        score_mode=args.score_mode,
    )
    _save_jsonl(remapped, args.output_queries_path, overwrite=args.overwrite)
    print(f"[Remap] Wrote remapped queries to {args.output_queries_path}")


if __name__ == "__main__":
    main()
