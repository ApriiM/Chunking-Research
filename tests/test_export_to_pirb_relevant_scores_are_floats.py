"""Validate exported PIRB query files use float values in relevant_scores."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import pytest


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def _export_root() -> Path:
    explicit_root = os.getenv("PIRB_FLOAT_SCORE_EXPORT_ROOT")
    if explicit_root:
        return Path(explicit_root)

    repo_root = Path(__file__).resolve().parents[1]
    preferred_root = repo_root / "export_to_pirb" / "Fixing_pirb_export_rerun_10m"
    if preferred_root.is_dir():
        return preferred_root
    return repo_root / "export_to_pirb"


def test_export_to_pirb_relevant_scores_are_floats() -> None:
    export_root = _export_root()
    if not export_root.is_dir():
        pytest.skip(f"Missing export root: {export_root}")

    queries_files = sorted(export_root.rglob("run_*/queries/queries.jsonl"))
    if not queries_files:
        pytest.skip(f"No exported queries files found under: {export_root}")

    violations: list[str] = []
    violation_limit = 50

    for queries_path in queries_files:
        for line_no, row in enumerate(_iter_jsonl(queries_path), start=1):
            query_id = str(row.get("id") or "")
            scores = row.get("relevant_scores")
            if not isinstance(scores, list):
                violations.append(
                    f"{queries_path}:{line_no} query_id={query_id} "
                    f"relevant_scores is not a list (type={type(scores).__name__})"
                )
                if len(violations) >= violation_limit:
                    break
                continue

            for idx, score in enumerate(scores):
                if not isinstance(score, float):
                    violations.append(
                        f"{queries_path}:{line_no} query_id={query_id} "
                        f"relevant_scores[{idx}]={score!r} type={type(score).__name__}"
                    )
                    if len(violations) >= violation_limit:
                        break
            if len(violations) >= violation_limit:
                break
        if len(violations) >= violation_limit:
            break

    assert not violations, (
        "Found non-float relevant_scores values in exported PIRB queries:\n"
        + "\n".join(violations)
    )
