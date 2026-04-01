"""Build per-dataset reports for queries with artificial relevant id '-1'."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MissingRelevantRow:
    """Aggregated `-1` relevant statistics for one exported dataset/run."""

    dataset: str
    run_name: str
    missing_count: int
    total_queries: int
    missing_percent: float
    queries_path: Path


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            yield json.loads(raw)


def _to_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def _read_dataset_name(run_dir: Path) -> str:
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dataset_slug = metadata.get("dataset_slug")
        if dataset_slug:
            return str(dataset_slug)
    return run_dir.name


def _count_missing_relevant(queries_path: Path) -> tuple[int, int]:
    missing_count = 0
    total_queries = 0

    for row in _iter_jsonl(queries_path):
        total_queries += 1
        relevant_values = _to_str_list(row.get("relevant"))
        if "-1" in relevant_values:
            missing_count += 1

    return missing_count, total_queries


def find_exported_run_dirs(input_root: Path) -> list[Path]:
    """Return run directories that contain exported PIRB `queries/queries.jsonl`."""

    run_dirs: list[Path] = []
    for run_dir in sorted(input_root.rglob("run_*")):
        if not run_dir.is_dir():
            continue
        queries_path = run_dir / "queries" / "queries.jsonl"
        if queries_path.exists():
            run_dirs.append(run_dir)
    return run_dirs


def build_missing_relevant_report(input_root: Path) -> list[MissingRelevantRow]:
    """Compute per-dataset counts and percentages for `relevant == '-1'` queries."""

    rows: list[MissingRelevantRow] = []
    for run_dir in find_exported_run_dirs(input_root):
        queries_path = run_dir / "queries" / "queries.jsonl"
        missing_count, total_queries = _count_missing_relevant(queries_path)
        missing_percent = 0.0
        if total_queries > 0:
            missing_percent = (missing_count / total_queries) * 100.0

        rows.append(
            MissingRelevantRow(
                dataset=_read_dataset_name(run_dir),
                run_name=run_dir.name,
                missing_count=missing_count,
                total_queries=total_queries,
                missing_percent=missing_percent,
                queries_path=queries_path,
            )
        )

    return rows


def render_report_tsv(rows: Sequence[MissingRelevantRow]) -> str:
    """Render report rows as TSV including percentage."""

    header = "dataset\trun\tmissing_count\ttotal_queries\tmissing_percent"
    lines = [header]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.dataset,
                    row.run_name,
                    str(row.missing_count),
                    str(row.total_queries),
                    f"{row.missing_percent:.2f}",
                ]
            )
        )
    return "\n".join(lines) + "\n"
