"""Tests for PIRB export baseline cache generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results_converter.baseline_cache import build_baseline_cache


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_run(export_session_path: Path, run_name: str, dataset_slug: str) -> Path:
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
    _write_jsonl(
        run_dir / "queries" / "queries.jsonl",
        [
            {"id": "q1", "relevant": ["0"], "relevant_scores": [1]},
            {"id": "q2", "relevant": ["-1"], "relevant_scores": [0.0]},
        ],
    )
    return run_dir


def test_build_baseline_cache_uses_only_finished_runs(tmp_path: Path) -> None:
    export_session = tmp_path / "export" / "session_abc"
    _write_run(export_session, "run_0001", "dataset_a")
    _write_run(export_session, "run_0002", "dataset_b")

    report_tsv = tmp_path / "report.tsv"
    report_tsv.write_text(
        "run\tstatus\texit_code\tduration_sec\tsuccessfully_converted\tfailed\tnote\tlog_path\n"
        "run_0001\tfinished\t0\t4\t1\t0\t\tlogs/run_0001.log\n"
        "run_0002\ttimed_out\t124\t60\tNA\tNA\t>60s\tlogs/run_0002.log\n",
        encoding="utf-8",
    )

    baseline_dir = build_baseline_cache(
        report_tsv_path=report_tsv,
        export_session_path=export_session,
        output_root=tmp_path / "baseline_out",
        baseline_name="baseline_unit",
    )

    manifest = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_count"] == 1
    assert manifest["runs"] == ["run_0001"]

    timing_lines = (baseline_dir / "timing_snapshot.tsv").read_text(encoding="utf-8").strip().splitlines()
    assert len(timing_lines) == 2
    assert timing_lines[1].startswith("run_0001\tdataset_a\tfinished\t4")

    signatures = (baseline_dir / "run_signatures.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(signatures) == 1
    payload = json.loads(signatures[0])
    assert payload["run_name"] == "run_0001"
    assert payload["dataset"] == "dataset_a"
    assert payload["query_count"] == 2
    assert payload["passage_count"] == 2
    assert payload["minus_one_query_count"] == 1
    assert payload["query_relevance_sha256"]
    assert payload["passage_parent_mapping_sha256"]
