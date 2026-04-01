"""Tests for indexed span lookup and query conversion edge cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results_converter.pirb_export import (
    _MergedPassageSpan,
    _convert_passages,
    _convert_queries,
    _find_covering_chunk_group,
)


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


def test_find_covering_chunk_group_with_indexed_spans() -> None:
    spans = [
        _MergedPassageSpan(passage_id="0", start=0, end=5, overlap_from_prev=0),
        _MergedPassageSpan(passage_id="1", start=3, end=10, overlap_from_prev=2),
        _MergedPassageSpan(passage_id="2", start=8, end=15, overlap_from_prev=2),
    ]
    span_starts = [span.start for span in spans]
    span_ends = [span.end for span in spans]

    groups_a = _find_covering_chunk_group(
        spans,
        span_starts=span_starts,
        span_ends=span_ends,
        span_start=4,
        span_end=9,
    )
    groups_b = _find_covering_chunk_group(
        spans,
        span_starts=span_starts,
        span_ends=span_ends,
        span_start=9,
        span_end=14,
    )
    groups_none = _find_covering_chunk_group(
        spans,
        span_starts=span_starts,
        span_ends=span_ends,
        span_start=15,
        span_end=16,
    )

    assert groups_a == [(0, 1), (1, 1)]
    assert groups_b == [(1, 2), (2, 2)]
    assert groups_none == []


def test_convert_queries_with_repeated_relevant_doc_ids(tmp_path: Path) -> None:
    src_passages = tmp_path / "run_x" / "passages.jsonl"
    dst_passages = tmp_path / "out" / "passages" / "passages.jsonl"
    src_queries = tmp_path / "run_x" / "queries" / "queries.jsonl"
    dst_queries = tmp_path / "out" / "queries" / "queries.jsonl"

    _write_jsonl(
        src_passages,
        [
            {"id": "pA", "contents": "Alpha Beta", "metadata": {"parentId": "d1"}},
            {"id": "pB", "contents": "Gamma Delta", "metadata": {"parentId": "d1"}},
        ],
    )
    _write_jsonl(
        src_queries,
        [
            {"id": "q_plain", "contents": "plain", "relevant": ["d1", "d1"]},
            {
                "id": "q_ext",
                "contents": "extractive",
                "relevant": ["d1", "d1"],
                "metadata": {"extractive_span_text_answer": ["Alpha"]},
            },
        ],
    )

    doc_to_passages, normalized_passages, _ = _convert_passages(
        src_passages, dst_passages
    )
    extractive_count, not_found_ids, _ = _convert_queries(
        src_queries,
        dst_queries,
        doc_to_passages=doc_to_passages,
        normalized_passage_contents=normalized_passages,
    )

    assert extractive_count == 1
    assert not_found_ids == []

    converted = _read_jsonl(dst_queries)
    plain_row = next(row for row in converted if row["id"] == "q_plain")
    ext_row = next(row for row in converted if row["id"] == "q_ext")

    assert plain_row["relevant"] == ["0", "1"]
    assert ext_row["relevant"] == ["0"]
