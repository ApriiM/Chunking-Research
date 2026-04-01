"""Tests for per-dataset reporting of queries containing relevant id '-1'."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results_converter.missing_relevant_report import (
    build_missing_relevant_report,
    render_report_tsv,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_metadata(run_dir: Path, dataset_slug: str) -> None:
    payload = {"dataset_slug": dataset_slug}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")


def test_build_missing_relevant_report_counts_per_dataset(tmp_path: Path) -> None:
    root = tmp_path / "export_root"

    run_1 = root / "exp" / "session_x" / "run_0001"
    run_2 = root / "exp" / "session_x" / "run_0002"

    _write_metadata(run_1, "dataset_a")
    _write_metadata(run_2, "dataset_b")

    _write_jsonl(
        run_1 / "queries" / "queries.jsonl",
        [
            {"id": "q1", "relevant": ["1", "2"]},
            {"id": "q2", "relevant": ["-1"]},
            {"id": "q3", "relevant": []},
        ],
    )
    _write_jsonl(
        run_2 / "queries" / "queries.jsonl",
        [
            {"id": "q4", "relevant": ["3"]},
            {"id": "q5", "relevant": ["4"]},
        ],
    )

    rows = build_missing_relevant_report(root)

    assert len(rows) == 2
    assert rows[0].dataset == "dataset_a"
    assert rows[0].missing_count == 1
    assert rows[0].total_queries == 3
    assert round(rows[0].missing_percent, 2) == 33.33

    assert rows[1].dataset == "dataset_b"
    assert rows[1].missing_count == 0
    assert rows[1].total_queries == 2
    assert rows[1].missing_percent == 0.0


def test_render_report_tsv_has_expected_columns(tmp_path: Path) -> None:
    root = tmp_path / "export_root"
    run = root / "exp" / "session_x" / "run_0003"
    _write_metadata(run, "dataset_c")
    _write_jsonl(run / "queries" / "queries.jsonl", [{"id": "q1", "relevant": ["-1"]}])

    rows = build_missing_relevant_report(root)
    report = render_report_tsv(rows)

    lines = report.strip().splitlines()
    assert lines[0] == "dataset\trun\tmissing_count\ttotal_queries\tmissing_percent"
    assert lines[1].startswith("dataset_c\trun_0003\t1\t1\t100.00")
