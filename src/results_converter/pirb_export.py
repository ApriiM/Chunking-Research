"""Export experiment runs to PIRB format with extractive relevance remapping."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Iterator, List, Optional, Sequence
from tqdm.auto import tqdm


REQUIRED_RUN_FILES: Sequence[str] = (
    "metadata.json",
    "passages.jsonl",
    "documents/documents.jsonl",
    "queries/queries.jsonl",
)


@dataclass(frozen=True)
class RunExportResult:
    source_run_dir: Path
    target_run_dir: Path
    copied_files: List[Path]


@dataclass(frozen=True)
class RunExportFailure:
    source_run_dir: Path
    reason: str


@dataclass(frozen=True)
class ExportSummary:
    input_path: Path
    output_root: Path
    successes: List[RunExportResult]
    failures: List[RunExportFailure]
    extractive_query_count: int
    extractive_not_found_ids: List[str]


@dataclass(frozen=True)
class PassageConversionStats:
    """Counts collected while converting one run's passages."""

    passages_count: int
    parent_doc_count: int


@dataclass(frozen=True)
class QueryConversionStats:
    """Timing and workload metrics collected while converting one run's queries."""

    total_queries: int
    extractive_query_count: int
    extractive_not_found_count: int
    unique_extractive_bins: int
    average_candidate_passages_per_bin: float
    max_candidate_passages_per_bin: int
    average_merged_chars_per_bin: float
    max_merged_chars_per_bin: int
    scan_phase_seconds: float
    prepare_bins_phase_seconds: float
    convert_phase_seconds: float


def _is_valid_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return all((path / rel).exists() for rel in REQUIRED_RUN_FILES)


def _missing_required_files(path: Path) -> List[str]:
    missing: List[str] = []
    for rel in REQUIRED_RUN_FILES:
        if not (path / rel).exists():
            missing.append(rel)
    return missing


def _iter_candidate_run_dirs(input_path: Path) -> Iterable[Path]:
    # Direct run folder case.
    if input_path.is_dir() and input_path.name.startswith("run_"):
        yield input_path

    # Recursive session / experiment folder case.
    for candidate in sorted(input_path.rglob("run_*")):
        if candidate.is_dir():
            yield candidate


def _discover_run_dirs(input_path: Path) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    dedup: List[Path] = []
    seen = set()
    for run_dir in _iter_candidate_run_dirs(input_path):
        resolved = run_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        dedup.append(resolved)

    if not dedup:
        raise FileNotFoundError(f"No run_* directories found under: {input_path}")

    return sorted(dedup)


def find_valid_run_dirs(input_path: Path) -> List[Path]:
    dedup = _discover_run_dirs(input_path)
    valid = [run_dir for run_dir in dedup if _is_valid_run_dir(run_dir)]
    if not valid:
        raise FileNotFoundError(
            "No valid run directories found. Each run must contain: "
            + ", ".join(REQUIRED_RUN_FILES)
        )
    return valid


def _relative_run_path(run_dir: Path, repo_root: Path, input_path: Path) -> Path:
    # Prefer full project-relative path when available.
    try:
        return run_dir.resolve().relative_to(repo_root.resolve())
    except ValueError:
        pass

    # Otherwise preserve relative shape under input path.
    try:
        return run_dir.resolve().relative_to(input_path.resolve())
    except ValueError:
        return Path(run_dir.name)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            yield json.loads(raw)


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _to_str_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    return [str(value)]


def _extract_parent_id(row: dict) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    parent_raw = (
        row.get("parentId")
        or row.get("parent_id")
        or row.get("document_id")
        or row.get("doc_id")
        or metadata.get("parentId")
        or metadata.get("parent_id")
        or metadata.get("document_id")
        or metadata.get("doc_id")
        or ""
    )
    return str(parent_raw)


def _convert_passages(
    src_passages: Path,
    dst_passages: Path,
) -> tuple[dict[str, List[str]], dict[str, str], PassageConversionStats]:
    doc_to_passages: dict[str, List[str]] = {}
    normalized_passage_contents: dict[str, str] = {}
    dst_passages.parent.mkdir(parents=True, exist_ok=True)
    passages_count = 0

    with dst_passages.open("w", encoding="utf-8") as out_f:
        for idx, row in enumerate(
            tqdm(
                _iter_jsonl(src_passages),
                desc=f"Converting passages ({src_passages.parent.name})",
                unit="passage",
            )
        ):
            passages_count += 1
            # Normalize exported passage IDs to contiguous ascending strings.
            pid = str(idx)
            original_id = str(row.get("id") or row.get("passage_id") or idx)
            contents = str(
                row.get("contents") or row.get("text") or row.get("content") or ""
            )
            parent_id = _extract_parent_id(row)

            converted_row = {
                "id": pid,
                "contents": contents,
                "metadata": {
                    "parentId": parent_id,
                    "original_id": original_id,
                },
            }
            out_f.write(json.dumps(converted_row, ensure_ascii=False) + "\n")
            normalized_passage_contents[pid] = _normalize_text_for_match(contents)
            mapped_parent_id = str(
                (converted_row.get("metadata") or {}).get("parentId") or ""
            )
            if mapped_parent_id:
                doc_to_passages.setdefault(mapped_parent_id, []).append(pid)

    return (
        doc_to_passages,
        normalized_passage_contents,
        PassageConversionStats(
            passages_count=passages_count,
            parent_doc_count=len(doc_to_passages),
        ),
    )


def _normalize_text_for_match(text: str) -> str:
    # Remove newlines, dots and spaces (incl. tabs) before substring matching.
    return re.sub(r"[\s\.]+", "", text or "")


def _find_overlap_len(left: str, right: str) -> int:
    if not left:
        return 0
    if not right:
        return 0

    max_overlap = min(len(left), len(right))
    for candidate in range(max_overlap, 0, -1):
        if left[-candidate:] == right[:candidate]:
            return candidate
    return 0


@dataclass(frozen=True)
class _MergedPassageSpan:
    passage_id: str
    start: int
    end: int
    # Number of chars skipped from this passage because they already existed
    # as suffix of the previously merged text.
    overlap_from_prev: int


@dataclass(frozen=True)
class _PreparedMergedBin:
    """Prepared merged-text cache for one extractive candidate bin."""

    candidate_passage_ids: List[str]
    merged_text: str
    spans: List[_MergedPassageSpan]
    span_starts: List[int]
    span_ends: List[int]


def _merge_all_passages_with_spans(
    candidate_passage_ids: List[str],
    normalized_passage_contents: dict[str, str],
) -> tuple[str, List[_MergedPassageSpan]]:
    merged = ""
    spans: List[_MergedPassageSpan] = []
    passage_iterator: Iterable[str]
    if len(candidate_passage_ids) >= 5000:
        passage_iterator = tqdm(
            candidate_passage_ids,
            desc="Merging passages inside bin",
            unit="passage",
            leave=False,
        )
    else:
        passage_iterator = candidate_passage_ids
    for pid in passage_iterator:
        passage_text = normalized_passage_contents.get(pid, "")
        overlap_len = _find_overlap_len(merged, passage_text)
        start = len(merged) - overlap_len
        end = start + len(passage_text)
        spans.append(
            _MergedPassageSpan(
                passage_id=pid,
                start=start,
                end=end,
                overlap_from_prev=overlap_len,
            )
        )
        merged += passage_text[overlap_len:]
    return merged, spans


def _iter_substring_positions(text: str, substring: str) -> Iterator[int]:
    start = 0
    while True:
        pos = text.find(substring, start)
        if pos < 0:
            return
        yield pos
        # Allow overlapping matches, e.g. "aaa" and "aa".
        start = pos + 1


def _find_covering_chunk_group(
    spans: List[_MergedPassageSpan],
    *,
    span_starts: Sequence[int],
    span_ends: Sequence[int],
    span_start: int,
    span_end: int,
) -> List[tuple[int, int]]:
    """Find all contiguous span groups covering [span_start, span_end)."""

    if not spans:
        return []

    # Candidate start index interval where:
    # span.start <= span_start < span.end
    first_idx_with_end_after_start = bisect_right(span_ends, span_start)
    last_idx_with_start_before_end = bisect_right(span_starts, span_start) - 1
    if first_idx_with_end_after_start > last_idx_with_start_before_end:
        return []

    groups: List[tuple[int, int]] = []
    for start_idx in range(
        first_idx_with_end_after_start,
        last_idx_with_start_before_end + 1,
    ):
        # First chunk index with chunk.end >= span_end.
        end_idx = bisect_left(span_ends, span_end, lo=start_idx)
        if end_idx >= len(spans):
            continue
        groups.append((start_idx, end_idx))
    return groups


def _candidate_passage_ids_for_docs(
    relevant_doc_ids: Sequence[str],
    doc_to_passages: dict[str, List[str]],
) -> List[str]:
    candidate_passage_ids: List[str] = []
    for doc_id in relevant_doc_ids:
        candidate_passage_ids.extend(doc_to_passages.get(doc_id, []))
    return _dedupe_preserve_order(candidate_passage_ids)


def _build_doc_key(relevant_doc_ids: Sequence[str]) -> tuple[str, ...]:
    """Return an order-preserving deduplicated key for relevant document IDs."""

    return tuple(_dedupe_preserve_order(relevant_doc_ids))


def _prepare_merged_bin(
    candidate_passage_ids: List[str],
    normalized_passage_contents: dict[str, str],
) -> _PreparedMergedBin:
    """Build merged text/spans and precompute indexes for fast span lookup."""

    merged_text, spans = _merge_all_passages_with_spans(
        candidate_passage_ids,
        normalized_passage_contents,
    )
    span_starts = [span.start for span in spans]
    span_ends = [span.end for span in spans]
    return _PreparedMergedBin(
        candidate_passage_ids=candidate_passage_ids,
        merged_text=merged_text,
        spans=spans,
        span_starts=span_starts,
        span_ends=span_ends,
    )


def _candidate_passages_for_doc_key_id(
    doc_key_id: int,
    *,
    doc_keys: Sequence[tuple[str, ...]],
    doc_to_passages: dict[str, List[str]],
    candidate_cache: dict[int, List[str]],
) -> List[str]:
    """Resolve candidate passage IDs for one doc-key id with memoization."""

    cached = candidate_cache.get(doc_key_id)
    if cached is not None:
        return cached
    candidate_passage_ids = _candidate_passage_ids_for_docs(
        doc_keys[doc_key_id],
        doc_to_passages,
    )
    candidate_cache[doc_key_id] = candidate_passage_ids
    return candidate_passage_ids


def _find_doc_key_id(
    relevant_doc_ids: Sequence[str],
    *,
    doc_key_to_id: dict[tuple[str, ...], int],
    doc_keys: List[tuple[str, ...]],
) -> int:
    """Intern a doc key and return its lightweight integer id."""

    doc_key = _build_doc_key(relevant_doc_ids)
    existing = doc_key_to_id.get(doc_key)
    if existing is not None:
        return existing
    new_id = len(doc_keys)
    doc_keys.append(doc_key)
    doc_key_to_id[doc_key] = new_id
    return new_id


def _find_extractive_relevant_passages_in_merged(
    candidate_passage_ids: List[str],
    merged_text: str,
    spans: List[_MergedPassageSpan],
    *,
    span_starts: Sequence[int],
    span_ends: Sequence[int],
    normalized_answers: List[str],
) -> tuple[List[str], List[float]]:
    if not candidate_passage_ids or not normalized_answers:
        return [], []
    if not merged_text:
        return [], []

    score_by_passage: dict[str, float] = {}
    normalized_answers = _dedupe_preserve_order(normalized_answers)
    for answer in normalized_answers:
        for span_start in _iter_substring_positions(merged_text, answer):
            span_end = span_start + len(answer)
            groups = _find_covering_chunk_group(
                spans,
                span_starts=span_starts,
                span_ends=span_ends,
                span_start=span_start,
                span_end=span_end,
            )
            for start_idx, end_idx in groups:
                group_size = (end_idx - start_idx) + 1
                score = 1.0 / float(group_size)
                for idx in range(start_idx, end_idx + 1):
                    pid = spans[idx].passage_id
                    previous = score_by_passage.get(pid, 0.0)
                    if score > previous:
                        score_by_passage[pid] = score

    relevant = [pid for pid in candidate_passage_ids if pid in score_by_passage]
    relevant_scores = [score_by_passage[pid] for pid in relevant]
    return relevant, relevant_scores


def _average_ints(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _convert_queries(
    src_queries: Path,
    dst_queries: Path,
    *,
    doc_to_passages: dict[str, List[str]],
    normalized_passage_contents: dict[str, str],
) -> tuple[int, List[str], QueryConversionStats]:
    dst_queries.parent.mkdir(parents=True, exist_ok=True)
    extractive_query_count = 0
    extractive_not_found_ids: List[str] = []
    total_queries = 0
    extractive_bin_ids: List[int] = []
    seen_extractive_bins: set[int] = set()
    candidate_passage_counts_per_bin: List[int] = []
    merged_chars_per_bin: List[int] = []
    doc_key_to_id: dict[tuple[str, ...], int] = {}
    doc_keys: List[tuple[str, ...]] = []
    candidate_passage_cache: dict[int, List[str]] = {}

    # First pass: count queries and group extractive queries by lightweight
    # doc-key bins so merged passage text is built once per unique candidate set.
    scan_phase_start = perf_counter()
    for row in tqdm(
        _iter_jsonl(src_queries),
        desc=f"Scanning queries ({src_queries.parent.parent.name})",
        unit="query",
    ):
        total_queries += 1
        metadata = row.get("metadata") or row.get("meta") or {}
        has_extractive = (
            isinstance(metadata, dict) and ("extractive_span_text_answer" in metadata)
        ) or ("extractive_span_text_answer" in row)
        if not has_extractive:
            continue

        relevant_doc_ids = _to_str_list(row.get("relevant"))
        doc_key_id = _find_doc_key_id(
            relevant_doc_ids,
            doc_key_to_id=doc_key_to_id,
            doc_keys=doc_keys,
        )
        if doc_key_id in seen_extractive_bins:
            continue

        seen_extractive_bins.add(doc_key_id)
        extractive_bin_ids.append(doc_key_id)
        candidate_passage_ids = _candidate_passages_for_doc_key_id(
            doc_key_id,
            doc_keys=doc_keys,
            doc_to_passages=doc_to_passages,
            candidate_cache=candidate_passage_cache,
        )
        candidate_passage_counts_per_bin.append(len(candidate_passage_ids))
    scan_phase_seconds = perf_counter() - scan_phase_start

    merged_cache: dict[int, _PreparedMergedBin] = {}
    prepare_bins_phase_start = perf_counter()
    for doc_key_id in tqdm(
        extractive_bin_ids,
        total=len(extractive_bin_ids),
        desc=f"Preparing extractive bins ({src_queries.parent.parent.name})",
        unit="bin",
    ):
        candidate_passage_ids = _candidate_passages_for_doc_key_id(
            doc_key_id,
            doc_keys=doc_keys,
            doc_to_passages=doc_to_passages,
            candidate_cache=candidate_passage_cache,
        )
        prepared = _prepare_merged_bin(
            candidate_passage_ids,
            normalized_passage_contents,
        )
        merged_cache[doc_key_id] = prepared
        merged_chars_per_bin.append(len(prepared.merged_text))
    prepare_bins_phase_seconds = perf_counter() - prepare_bins_phase_start

    convert_phase_start = perf_counter()
    with dst_queries.open("w", encoding="utf-8") as out_f:
        for idx, row in enumerate(
            tqdm(
                _iter_jsonl(src_queries),
                total=total_queries,
                desc=f"Converting queries ({src_queries.parent.parent.name})",
                unit="query",
            )
        ):
            qid = str(row.get("id") or row.get("query_id") or idx)
            contents = str(
                row.get("contents")
                or row.get("query")
                or row.get("text")
                or row.get("question")
                or ""
            )
            # Source relevant is expected to be document IDs.
            relevant_doc_ids = _to_str_list(row.get("relevant"))
            doc_key_id = _find_doc_key_id(
                relevant_doc_ids,
                doc_key_to_id=doc_key_to_id,
                doc_keys=doc_keys,
            )
            metadata = row.get("metadata") or row.get("meta") or {}
            has_extractive = (
                isinstance(metadata, dict)
                and ("extractive_span_text_answer" in metadata)
            ) or ("extractive_span_text_answer" in row)
            candidate_passage_ids = _candidate_passages_for_doc_key_id(
                doc_key_id,
                doc_keys=doc_keys,
                doc_to_passages=doc_to_passages,
                candidate_cache=candidate_passage_cache,
            )

            if has_extractive:
                extractive_query_count += 1
                extractive_values: object = (
                    metadata.get("extractive_span_text_answer")
                    if isinstance(metadata, dict)
                    else None
                )
                if extractive_values is None:
                    extractive_values = row.get("extractive_span_text_answer")
                normalized_answers = [
                    _normalize_text_for_match(answer)
                    for answer in _to_str_list(extractive_values)
                ]
                normalized_answers = [answer for answer in normalized_answers if answer]

                prepared = merged_cache.get(doc_key_id)
                if prepared is None:
                    prepared = _prepare_merged_bin(
                        candidate_passage_ids,
                        normalized_passage_contents,
                    )
                relevant, relevant_scores = (
                    _find_extractive_relevant_passages_in_merged(
                        prepared.candidate_passage_ids,
                        prepared.merged_text,
                        prepared.spans,
                        span_starts=prepared.span_starts,
                        span_ends=prepared.span_ends,
                        normalized_answers=normalized_answers,
                    )
                )
                if not relevant:
                    extractive_not_found_ids.append(qid)
            else:
                relevant = candidate_passage_ids
                relevant_scores = [1] * len(relevant)

            converted_row = {
                "id": qid,
                "contents": contents,
                "relevant": relevant,
                "relevant_scores": relevant_scores,
            }
            export_metadata: dict[str, object] = {}
            if isinstance(metadata, dict):
                if "free_text_answer" in metadata:
                    export_metadata["free_text_answer"] = metadata["free_text_answer"]
                if "extractive_span_text_answer" in metadata:
                    export_metadata["extractive_span_text_answer"] = metadata[
                        "extractive_span_text_answer"
                    ]
            if "free_text_answer" in row and "free_text_answer" not in export_metadata:
                export_metadata["free_text_answer"] = row["free_text_answer"]
            if (
                "extractive_span_text_answer" in row
                and "extractive_span_text_answer" not in export_metadata
            ):
                export_metadata["extractive_span_text_answer"] = row[
                    "extractive_span_text_answer"
                ]
            if export_metadata:
                converted_row["metadata"] = export_metadata
            out_f.write(json.dumps(converted_row, ensure_ascii=False) + "\n")
    convert_phase_seconds = perf_counter() - convert_phase_start

    stats = QueryConversionStats(
        total_queries=total_queries,
        extractive_query_count=extractive_query_count,
        extractive_not_found_count=len(extractive_not_found_ids),
        unique_extractive_bins=len(extractive_bin_ids),
        average_candidate_passages_per_bin=_average_ints(
            candidate_passage_counts_per_bin
        ),
        max_candidate_passages_per_bin=max(candidate_passage_counts_per_bin, default=0),
        average_merged_chars_per_bin=_average_ints(merged_chars_per_bin),
        max_merged_chars_per_bin=max(merged_chars_per_bin, default=0),
        scan_phase_seconds=scan_phase_seconds,
        prepare_bins_phase_seconds=prepare_bins_phase_seconds,
        convert_phase_seconds=convert_phase_seconds,
    )

    return extractive_query_count, extractive_not_found_ids, stats


def _ensure_non_empty_relevant_in_queries(queries_path: Path) -> int:
    rows = list(
        tqdm(
            _iter_jsonl(queries_path),
            desc=f"Post-check empty relevant ({queries_path.parent.parent.name})",
            unit="query",
        )
    )
    patched = 0
    for row in rows:
        relevant = row.get("relevant")
        if isinstance(relevant, list):
            if relevant:
                continue
        elif relevant:
            continue

        row["relevant"] = ["-1"]
        row["relevant_scores"] = [0.0]
        patched += 1

    if patched == 0:
        return 0

    with queries_path.open("w", encoding="utf-8") as out_f:
        for row in tqdm(
            rows,
            total=len(rows),
            desc=f"Writing patched queries ({queries_path.parent.parent.name})",
            unit="query",
        ):
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return patched


def export_runs_to_pirb(
    input_path: Path,
    output_root: Path,
    *,
    overwrite_run_dir: bool = False,
    repo_root: Path | None = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> ExportSummary:
    repo_root = (repo_root or Path.cwd()).resolve()
    input_path = input_path.resolve()
    output_root = output_root.resolve()

    run_dirs = _discover_run_dirs(input_path)
    results: List[RunExportResult] = []
    failures: List[RunExportFailure] = []
    extractive_query_count = 0
    extractive_not_found_ids: List[str] = []

    total_runs = len(run_dirs)
    for idx, run_dir in enumerate(run_dirs, start=1):
        if log_fn:
            log_fn(f"[RUN {idx}/{total_runs}] [START] processing {run_dir}")
        run_start = perf_counter()
        missing = _missing_required_files(run_dir)
        if missing:
            reason = "missing required file(s): " + ", ".join(missing)
            failures.append(RunExportFailure(source_run_dir=run_dir, reason=reason))
            if log_fn:
                log_fn(f"[FAIL] {run_dir} -> {reason}")
            continue

        rel_run = _relative_run_path(
            run_dir, repo_root=repo_root, input_path=input_path
        )
        target_run_dir = output_root / rel_run

        try:
            if target_run_dir.exists() and overwrite_run_dir:
                shutil.rmtree(target_run_dir)
            target_run_dir.mkdir(parents=True, exist_ok=True)

            copied: List[Path] = []

            # Copy metadata at run root.
            src_metadata = run_dir / "metadata.json"
            dst_metadata = target_run_dir / "metadata.json"
            if log_fn:
                log_fn("  [START] copying metadata.json")
            _copy_file(src_metadata, dst_metadata)
            copied.append(dst_metadata)
            if log_fn:
                log_fn("  [DONE] copying metadata.json")

            # Convert passages under passages/.
            src_passages = run_dir / "passages.jsonl"
            dst_passages = target_run_dir / "passages" / "passages.jsonl"
            if log_fn:
                log_fn("  [START] converting passages.jsonl")
            passages_phase_start = perf_counter()
            (
                doc_to_passages,
                normalized_passage_contents,
                passages_stats,
            ) = _convert_passages(
                src_passages,
                dst_passages,
            )
            passages_phase_seconds = perf_counter() - passages_phase_start
            copied.append(dst_passages)
            if log_fn:
                log_fn("  [DONE] converting passages.jsonl")

            # Convert queries under queries/.
            src_queries = run_dir / "queries" / "queries.jsonl"
            dst_queries = target_run_dir / "queries" / "queries.jsonl"
            if log_fn:
                log_fn("  [START] converting queries/queries.jsonl")
            queries_phase_start = perf_counter()
            (
                run_extractive_count,
                run_extractive_not_found,
                query_stats,
            ) = _convert_queries(
                src_queries,
                dst_queries,
                doc_to_passages=doc_to_passages,
                normalized_passage_contents=normalized_passage_contents,
            )
            queries_phase_seconds = perf_counter() - queries_phase_start
            copied.append(dst_queries)
            if log_fn:
                log_fn("  [DONE] converting queries/queries.jsonl")
            extractive_query_count += run_extractive_count
            extractive_not_found_ids.extend(
                [f"{run_dir} | {query_id}" for query_id in run_extractive_not_found]
            )

            results.append(
                RunExportResult(
                    source_run_dir=run_dir,
                    target_run_dir=target_run_dir,
                    copied_files=copied,
                )
            )
            if log_fn:
                log_fn(
                    "  [TIMING] "
                    f"passages_seconds={passages_phase_seconds:.3f} "
                    f"queries_seconds={queries_phase_seconds:.3f} "
                    f"run_total_seconds={(perf_counter() - run_start):.3f}"
                )
                log_fn(
                    "  [PASSAGES] "
                    f"passages_count={passages_stats.passages_count} "
                    f"parent_doc_count={passages_stats.parent_doc_count}"
                )
                log_fn(
                    "  [QUERIES] "
                    f"total_queries={query_stats.total_queries} "
                    f"extractive_queries={query_stats.extractive_query_count} "
                    f"extractive_not_found={query_stats.extractive_not_found_count} "
                    f"unique_extractive_bins={query_stats.unique_extractive_bins} "
                    f"avg_bin_passages={query_stats.average_candidate_passages_per_bin:.2f} "
                    f"max_bin_passages={query_stats.max_candidate_passages_per_bin} "
                    f"avg_bin_chars={query_stats.average_merged_chars_per_bin:.2f} "
                    f"max_bin_chars={query_stats.max_merged_chars_per_bin}"
                )
                log_fn(
                    "  [QUERIES_TIMING] "
                    f"scan_seconds={query_stats.scan_phase_seconds:.3f} "
                    f"prepare_bins_seconds={query_stats.prepare_bins_phase_seconds:.3f} "
                    f"convert_seconds={query_stats.convert_phase_seconds:.3f}"
                )
                log_fn(f"  [DONE] run conversion -> {target_run_dir}")
                log_fn(
                    f"[RUN {idx}/{total_runs}] [DONE] processing {run_dir} "
                    f"(seconds={(perf_counter() - run_start):.3f})"
                )
        except Exception as exc:
            reason = f"copy failed: {type(exc).__name__}: {exc}"
            failures.append(RunExportFailure(source_run_dir=run_dir, reason=reason))
            if log_fn:
                log_fn(f"[FAIL] {run_dir} -> {reason}")

    patched_queries = 0
    if log_fn:
        log_fn("[POST] [START] patching empty relevant lists in exported queries")
    for result in results:
        queries_path = result.target_run_dir / "queries" / "queries.jsonl"
        if not queries_path.is_file():
            continue
        patched_queries += _ensure_non_empty_relevant_in_queries(queries_path)
    if log_fn:
        log_fn(
            f"[POST] [DONE] patching empty relevant lists in exported queries: "
            f"{patched_queries}"
        )

    return ExportSummary(
        input_path=input_path,
        output_root=output_root,
        successes=results,
        failures=failures,
        extractive_query_count=extractive_query_count,
        extractive_not_found_ids=extractive_not_found_ids,
    )
