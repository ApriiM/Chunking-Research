from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence


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


def _convert_passages(
    src_passages: Path,
    dst_passages: Path,
) -> dict[str, List[str]]:
    doc_to_passages: dict[str, List[str]] = {}
    dst_passages.parent.mkdir(parents=True, exist_ok=True)

    with dst_passages.open("w", encoding="utf-8") as out_f:
        for idx, row in enumerate(_iter_jsonl(src_passages)):
            # Normalize exported passage IDs to contiguous ascending strings.
            pid = str(idx)
            contents = str(
                row.get("contents")
                or row.get("text")
                or row.get("content")
                or ""
            )
            parent_id = str(
                row.get("parentId")
                or row.get("parent_id")
                or row.get("document_id")
                or row.get("doc_id")
                or ""
            )

            converted_row = {
                "id": pid,
                "contents": contents,
            }
            out_f.write(json.dumps(converted_row, ensure_ascii=False) + "\n")
            if parent_id:
                doc_to_passages.setdefault(parent_id, []).append(pid)

    return doc_to_passages


def _convert_queries(
    src_queries: Path,
    dst_queries: Path,
    *,
    doc_to_passages: dict[str, List[str]],
) -> None:
    dst_queries.parent.mkdir(parents=True, exist_ok=True)
    with dst_queries.open("w", encoding="utf-8") as out_f:
        for idx, row in enumerate(_iter_jsonl(src_queries)):
            qid = str(row.get("id") or row.get("query_id") or idx)
            contents = str(
                row.get("contents")
                or row.get("query")
                or row.get("text")
                or row.get("question")
                or ""
            )
            relevant = _to_str_list(row.get("relevant"))
            metadata = row.get("metadata") or row.get("meta") or {}
            has_extractive = (
                (isinstance(metadata, dict) and ("extractive_span_text_answer" in metadata))
                or ("extractive_span_text_answer" in row)
            )

            if not has_extractive:
                expanded: List[str] = []
                for doc_id in relevant:
                    expanded.extend(doc_to_passages.get(doc_id, []))
                relevant = _dedupe_preserve_order(expanded)

            converted_row = {
                "id": qid,
                "contents": contents,
                "relevant": relevant,
                "relevant_scores": [1] * len(relevant),
            }
            out_f.write(json.dumps(converted_row, ensure_ascii=False) + "\n")


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

    total_runs = len(run_dirs)
    for idx, run_dir in enumerate(run_dirs, start=1):
        if log_fn:
            log_fn(f"[RUN {idx}/{total_runs}] processing {run_dir}")
        missing = _missing_required_files(run_dir)
        if missing:
            reason = "missing required file(s): " + ", ".join(missing)
            failures.append(RunExportFailure(source_run_dir=run_dir, reason=reason))
            if log_fn:
                log_fn(f"[FAIL] {run_dir} -> {reason}")
            continue

        rel_run = _relative_run_path(run_dir, repo_root=repo_root, input_path=input_path)
        target_run_dir = output_root / rel_run

        try:
            if target_run_dir.exists() and overwrite_run_dir:
                shutil.rmtree(target_run_dir)
            target_run_dir.mkdir(parents=True, exist_ok=True)

            copied: List[Path] = []

            # Copy metadata at run root.
            src_metadata = run_dir / "metadata.json"
            dst_metadata = target_run_dir / "metadata.json"
            _copy_file(src_metadata, dst_metadata)
            copied.append(dst_metadata)

            # Convert passages under passages/.
            src_passages = run_dir / "passages.jsonl"
            dst_passages = target_run_dir / "passages" / "passages.jsonl"
            if log_fn:
                log_fn("  - converting passages.jsonl")
            doc_to_passages = _convert_passages(
                src_passages,
                dst_passages,
            )
            copied.append(dst_passages)

            # Convert queries under queries/.
            src_queries = run_dir / "queries" / "queries.jsonl"
            dst_queries = target_run_dir / "queries" / "queries.jsonl"
            if log_fn:
                log_fn("  - converting queries/queries.jsonl")
            _convert_queries(
                src_queries,
                dst_queries,
                doc_to_passages=doc_to_passages,
            )
            copied.append(dst_queries)

            results.append(
                RunExportResult(
                    source_run_dir=run_dir,
                    target_run_dir=target_run_dir,
                    copied_files=copied,
                )
            )
            if log_fn:
                log_fn(f"  -> OK {target_run_dir}")
        except Exception as exc:
            reason = f"copy failed: {type(exc).__name__}: {exc}"
            failures.append(RunExportFailure(source_run_dir=run_dir, reason=reason))
            if log_fn:
                log_fn(f"[FAIL] {run_dir} -> {reason}")

    return ExportSummary(
        input_path=input_path,
        output_root=output_root,
        successes=results,
        failures=failures,
    )
