"""Tests for PIRB export stage metrics and workload counters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results_converter.pirb_export import _convert_passages, _convert_queries


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def test_convert_passages_reports_counts(tmp_path: Path) -> None:
    src_passages = tmp_path / "run_x" / "passages.jsonl"
    dst_passages = tmp_path / "out" / "passages" / "passages.jsonl"

    _write_jsonl(
        src_passages,
        [
            {"id": "pA", "contents": "Alpha Beta", "metadata": {"parentId": "d1"}},
            {"id": "pB", "contents": "Gamma", "metadata": {"parentId": "d1"}},
            {"id": "pC", "contents": "Delta", "metadata": {"parentId": "d2"}},
        ],
    )

    doc_to_passages, normalized, stats = _convert_passages(src_passages, dst_passages)

    assert stats.passages_count == 3
    assert stats.parent_doc_count == 2
    assert doc_to_passages["d1"] == ["0", "1"]
    assert doc_to_passages["d2"] == ["2"]
    assert normalized["0"] == "AlphaBeta"


def test_convert_queries_reports_phase_metrics(tmp_path: Path) -> None:
    src_passages = tmp_path / "run_x" / "passages.jsonl"
    dst_passages = tmp_path / "out" / "passages" / "passages.jsonl"
    src_queries = tmp_path / "run_x" / "queries" / "queries.jsonl"
    dst_queries = tmp_path / "out" / "queries" / "queries.jsonl"

    _write_jsonl(
        src_passages,
        [
            {"id": "pA", "contents": "Alpha Beta", "metadata": {"parentId": "d1"}},
            {"id": "pB", "contents": "Gamma", "metadata": {"parentId": "d1"}},
        ],
    )
    _write_jsonl(
        src_queries,
        [
            {"id": "q1", "contents": "plain", "relevant": ["d1"]},
            {
                "id": "q2",
                "contents": "extractive hit",
                "relevant": ["d1"],
                "metadata": {"extractive_span_text_answer": ["Alpha"]},
            },
            {
                "id": "q3",
                "contents": "extractive miss",
                "relevant": ["d1"],
                "metadata": {"extractive_span_text_answer": ["NotFound"]},
            },
        ],
    )

    doc_to_passages, normalized, _ = _convert_passages(src_passages, dst_passages)
    extractive_count, not_found_ids, stats = _convert_queries(
        src_queries,
        dst_queries,
        doc_to_passages=doc_to_passages,
        normalized_passage_contents=normalized,
    )

    assert extractive_count == 2
    assert not_found_ids == ["q3"]
    assert stats.total_queries == 3
    assert stats.extractive_query_count == 2
    assert stats.extractive_not_found_count == 1
    assert stats.unique_extractive_bins == 1
    assert stats.average_candidate_passages_per_bin == 2.0
    assert stats.max_candidate_passages_per_bin == 2
    assert stats.max_merged_chars_per_bin > 0
    assert stats.scan_phase_seconds >= 0.0
    assert stats.prepare_bins_phase_seconds >= 0.0
    assert stats.convert_phase_seconds >= 0.0

    converted_rows = _read_jsonl(dst_queries)
    assert len(converted_rows) == 3
    q1_row = next(row for row in converted_rows if row["id"] == "q1")
    q2_row = next(row for row in converted_rows if row["id"] == "q2")
    q3_row = next(row for row in converted_rows if row["id"] == "q3")
    assert q1_row["relevant_scores"]
    assert all(isinstance(score, float) for score in q1_row["relevant_scores"])
    assert set(q1_row["relevant_scores"]) == {1.0}
    assert q2_row["relevant"]
    assert q3_row["relevant"] == []
