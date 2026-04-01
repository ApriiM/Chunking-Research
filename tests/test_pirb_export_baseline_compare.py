"""Tests for baseline-to-current PIRB export comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results_converter.baseline_cache import build_baseline_cache
from src.results_converter.baseline_compare import compare_against_baseline


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_run(
    export_session_path: Path,
    run_name: str,
    dataset_slug: str,
    *,
    use_minus_one: bool,
) -> None:
    run_dir = export_session_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"dataset_slug": dataset_slug}),
        encoding="utf-8",
    )
    _write_jsonl(
        run_dir / "passages" / "passages.jsonl",
        [
            {
                "id": "0",
                "contents": "alpha",
                "metadata": {"parentId": "doc-1", "original_id": "orig-0"},
            },
            {
                "id": "1",
                "contents": "beta",
                "metadata": {"parentId": "doc-1", "original_id": "orig-1"},
            },
        ],
    )
    relevant_list = ["-1"] if use_minus_one else ["0"]
    _write_jsonl(
        run_dir / "queries" / "queries.jsonl",
        [
            {"id": "q1", "relevant": relevant_list, "relevant_scores": [1]},
            {"id": "q2", "relevant": ["1"], "relevant_scores": [1]},
        ],
    )


def _write_report(path: Path, duration_sec: int) -> None:
    path.write_text(
        "run\tstatus\texit_code\tduration_sec\tsuccessfully_converted\tfailed\tnote\tlog_path\n"
        f"run_0001\tfinished\t0\t{duration_sec}\t1\t0\t\tlogs/run_0001.log\n",
        encoding="utf-8",
    )


def test_compare_against_baseline_passes_when_outputs_match(tmp_path: Path) -> None:
    baseline_export = tmp_path / "baseline_export" / "session_a"
    _write_run(baseline_export, "run_0001", "dataset_a", use_minus_one=False)
    baseline_report = tmp_path / "baseline_report.tsv"
    _write_report(baseline_report, duration_sec=5)

    baseline_dir = build_baseline_cache(
        report_tsv_path=baseline_report,
        export_session_path=baseline_export,
        output_root=tmp_path / "baseline_cache",
        baseline_name="baseline_unit",
    )

    current_export = tmp_path / "current_export" / "session_a"
    _write_run(current_export, "run_0001", "dataset_a", use_minus_one=False)
    current_report = tmp_path / "current_report.tsv"
    _write_report(current_report, duration_sec=5)

    summary = compare_against_baseline(
        baseline_dir=baseline_dir,
        export_session_path=current_export,
        candidate_report_tsv_path=current_report,
        max_duration_delta_sec=0,
    )

    assert summary.overall_passed
    assert summary.all_signatures_match
    assert summary.all_timings_within_limit
    assert summary.signature_match_count == 1
    assert summary.timing_checked_count == 1
    assert summary.runs[0].duration_delta_sec == 0


def test_compare_against_baseline_detects_mismatch_and_timing_delta(
    tmp_path: Path,
) -> None:
    baseline_export = tmp_path / "baseline_export" / "session_b"
    _write_run(baseline_export, "run_0001", "dataset_b", use_minus_one=False)
    baseline_report = tmp_path / "baseline_report.tsv"
    _write_report(baseline_report, duration_sec=4)

    baseline_dir = build_baseline_cache(
        report_tsv_path=baseline_report,
        export_session_path=baseline_export,
        output_root=tmp_path / "baseline_cache",
        baseline_name="baseline_unit",
    )

    current_export = tmp_path / "current_export" / "session_b"
    _write_run(current_export, "run_0001", "dataset_b", use_minus_one=True)
    current_report = tmp_path / "current_report.tsv"
    _write_report(current_report, duration_sec=12)

    summary = compare_against_baseline(
        baseline_dir=baseline_dir,
        export_session_path=current_export,
        candidate_report_tsv_path=current_report,
        max_duration_delta_sec=2,
    )

    assert not summary.overall_passed
    assert not summary.all_signatures_match
    assert not summary.all_timings_within_limit
    assert summary.runs[0].signature_match is False
    assert "minus_one_query_count" in summary.runs[0].mismatched_signature_fields
    assert summary.runs[0].duration_delta_sec == 8
    assert summary.runs[0].duration_within_limit is False
