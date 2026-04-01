#!/usr/bin/env python3
"""Compare exported PIRB runs against baseline signatures and timing snapshots."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results_converter.baseline_compare import (
    compare_against_baseline,
    write_comparison_report,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for baseline comparison."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare current export outputs with baseline signatures and "
            "optionally compare per-run durations."
        )
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Path to baseline cache directory (contains run_signatures.jsonl).",
    )
    parser.add_argument(
        "--export-session-path",
        required=True,
        help="Path to current export session containing run_* directories.",
    )
    parser.add_argument(
        "--candidate-report-tsv",
        default="",
        help="Optional per-run TSV report to compare duration deltas.",
    )
    parser.add_argument(
        "--max-duration-delta-sec",
        type=int,
        default=2,
        help="Allowed absolute per-run duration difference in seconds.",
    )
    parser.add_argument(
        "--output-path",
        default="",
        help="Output JSON report path; defaults to baseline_dir/comparison_*.json",
    )
    parser.add_argument(
        "--allow-diff",
        action="store_true",
        help="Return exit code 0 even when comparison fails.",
    )
    return parser.parse_args()


def main() -> None:
    """Run baseline comparison and write report."""

    args = parse_args()

    candidate_report_tsv_path: Path | None = None
    if args.candidate_report_tsv:
        candidate_report_tsv_path = Path(args.candidate_report_tsv)

    summary = compare_against_baseline(
        baseline_dir=Path(args.baseline_dir),
        export_session_path=Path(args.export_session_path),
        candidate_report_tsv_path=candidate_report_tsv_path,
        max_duration_delta_sec=args.max_duration_delta_sec,
    )

    output_path = Path(args.output_path) if args.output_path else (
        Path(args.baseline_dir)
        / datetime.now(timezone.utc).strftime("comparison_%Y%m%dT%H%M%SZ.json")
    )
    written = write_comparison_report(summary, output_path)

    print(
        "overall_passed="
        f"{summary.overall_passed} "
        f"signatures={summary.signature_match_count}/{summary.run_count} "
        f"timing={summary.timing_within_limit_count}/{summary.timing_checked_count}"
    )
    print(written)

    if not summary.overall_passed and not args.allow_diff:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
