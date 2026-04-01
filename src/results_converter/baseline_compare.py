"""Compare PIRB export outputs against a previously captured baseline cache."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.results_converter.baseline_cache import (
    RunSignature,
    collect_run_signature,
    parse_timing_report,
)


SIGNATURE_FIELDS: tuple[str, ...] = (
    "dataset",
    "query_count",
    "passage_count",
    "minus_one_query_count",
    "query_id_order_sha256",
    "query_relevance_sha256",
    "passage_parent_mapping_sha256",
    "queries_file_sha256",
    "passages_file_sha256",
)


@dataclass(frozen=True)
class RunComparison:
    """Comparison result for one run against the baseline snapshot."""

    run_name: str
    signature_match: bool
    mismatched_signature_fields: list[str]
    baseline_duration_sec: int | None
    current_duration_sec: int | None
    duration_delta_sec: int | None
    duration_within_limit: bool | None


@dataclass(frozen=True)
class ComparisonSummary:
    """Aggregated baseline-vs-current comparison result."""

    generated_at_utc: str
    baseline_dir: str
    export_session_path: str
    candidate_report_tsv_path: str | None
    max_duration_delta_sec: int
    run_count: int
    signature_match_count: int
    timing_checked_count: int
    timing_within_limit_count: int
    all_signatures_match: bool
    all_timings_within_limit: bool
    overall_passed: bool
    runs: list[RunComparison]


def _read_jsonl(path: Path) -> list[dict]:
    payloads: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            payloads.append(json.loads(line))
    return payloads


def _load_baseline_signatures(baseline_dir: Path) -> dict[str, RunSignature]:
    signatures_path = baseline_dir / "run_signatures.jsonl"
    payloads = _read_jsonl(signatures_path)

    by_run: dict[str, RunSignature] = {}
    for row in payloads:
        run_name = str(row["run_name"])
        by_run[run_name] = RunSignature(
            run_name=run_name,
            dataset=str(row.get("dataset") or ""),
            queries_path=str(row.get("queries_path") or ""),
            passages_path=str(row.get("passages_path") or ""),
            query_count=int(row.get("query_count") or 0),
            passage_count=int(row.get("passage_count") or 0),
            minus_one_query_count=int(row.get("minus_one_query_count") or 0),
            query_id_order_sha256=str(row.get("query_id_order_sha256") or ""),
            query_relevance_sha256=str(row.get("query_relevance_sha256") or ""),
            passage_parent_mapping_sha256=str(
                row.get("passage_parent_mapping_sha256") or ""
            ),
            queries_file_sha256=str(row.get("queries_file_sha256") or ""),
            passages_file_sha256=str(row.get("passages_file_sha256") or ""),
        )
    return by_run


def _load_baseline_timings(baseline_dir: Path) -> dict[str, int]:
    timing_path = baseline_dir / "timing_snapshot.tsv"
    by_run: dict[str, int] = {}
    with timing_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            run_name = str(row.get("run") or "")
            by_run[run_name] = int(row.get("duration_sec") or 0)
    return by_run


def _compare_signature_fields(
    baseline_signature: RunSignature,
    current_signature: RunSignature,
) -> list[str]:
    mismatched: list[str] = []
    for field in SIGNATURE_FIELDS:
        if getattr(baseline_signature, field) != getattr(current_signature, field):
            mismatched.append(field)
    return mismatched


def compare_against_baseline(
    baseline_dir: Path,
    export_session_path: Path,
    *,
    candidate_report_tsv_path: Path | None = None,
    max_duration_delta_sec: int = 2,
) -> ComparisonSummary:
    """Compare current exported runs against baseline signatures and timings."""

    baseline_signatures = _load_baseline_signatures(baseline_dir)
    baseline_timings = _load_baseline_timings(baseline_dir)

    candidate_timings: dict[str, int] = {}
    if candidate_report_tsv_path is not None:
        for record in parse_timing_report(candidate_report_tsv_path):
            candidate_timings[record.run_name] = record.duration_sec

    run_names = sorted(baseline_signatures.keys())
    run_results: list[RunComparison] = []

    signature_match_count = 0
    timing_checked_count = 0
    timing_within_limit_count = 0

    for run_name in run_names:
        baseline_signature = baseline_signatures[run_name]
        current_signature = collect_run_signature(export_session_path / run_name)
        mismatched_fields = _compare_signature_fields(
            baseline_signature,
            current_signature,
        )
        signature_match = len(mismatched_fields) == 0
        if signature_match:
            signature_match_count += 1

        baseline_duration_sec = baseline_timings.get(run_name)
        current_duration_sec = candidate_timings.get(run_name)
        duration_delta_sec: int | None = None
        duration_within_limit: bool | None = None
        if baseline_duration_sec is not None and current_duration_sec is not None:
            duration_delta_sec = current_duration_sec - baseline_duration_sec
            duration_within_limit = (
                abs(duration_delta_sec) <= max_duration_delta_sec
            )
            timing_checked_count += 1
            if duration_within_limit:
                timing_within_limit_count += 1

        run_results.append(
            RunComparison(
                run_name=run_name,
                signature_match=signature_match,
                mismatched_signature_fields=mismatched_fields,
                baseline_duration_sec=baseline_duration_sec,
                current_duration_sec=current_duration_sec,
                duration_delta_sec=duration_delta_sec,
                duration_within_limit=duration_within_limit,
            )
        )

    all_signatures_match = signature_match_count == len(run_results)
    all_timings_within_limit = timing_checked_count == timing_within_limit_count
    overall_passed = all_signatures_match and all_timings_within_limit

    return ComparisonSummary(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        baseline_dir=str(baseline_dir),
        export_session_path=str(export_session_path),
        candidate_report_tsv_path=(
            str(candidate_report_tsv_path)
            if candidate_report_tsv_path is not None
            else None
        ),
        max_duration_delta_sec=max_duration_delta_sec,
        run_count=len(run_results),
        signature_match_count=signature_match_count,
        timing_checked_count=timing_checked_count,
        timing_within_limit_count=timing_within_limit_count,
        all_signatures_match=all_signatures_match,
        all_timings_within_limit=all_timings_within_limit,
        overall_passed=overall_passed,
        runs=run_results,
    )


def write_comparison_report(summary: ComparisonSummary, output_path: Path) -> Path:
    """Write comparison summary to JSON and return output path."""

    payload = asdict(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
