#!/usr/bin/env python3
"""Create baseline cache artifacts for PIRB export regression validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results_converter.baseline_cache import build_baseline_cache


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for baseline cache generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Build baseline signatures and timing snapshots from an export session "
            "using finished rows from a batch timing report."
        )
    )
    parser.add_argument(
        "--report-tsv",
        required=True,
        help="Path to per-run TSV report (e.g. pirb_per_run_*.tsv).",
    )
    parser.add_argument(
        "--export-session-path",
        required=True,
        help="Path to exported session directory containing run_* folders.",
    )
    parser.add_argument(
        "--output-root",
        default="export_to_pirb/Fixing_pirb_export/_baseline_cache",
        help="Output root where baseline folder will be created.",
    )
    parser.add_argument(
        "--baseline-name",
        default="",
        help="Optional baseline folder name. If empty, a UTC timestamped name is used.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate baseline cache directory and print created path."""

    args = parse_args()
    baseline_name = args.baseline_name
    if not baseline_name:
        baseline_name = datetime.now(timezone.utc).strftime("baseline_%Y%m%dT%H%M%SZ")

    baseline_dir = build_baseline_cache(
        report_tsv_path=Path(args.report_tsv),
        export_session_path=Path(args.export_session_path),
        output_root=Path(args.output_root),
        baseline_name=baseline_name,
    )
    print(baseline_dir)


if __name__ == "__main__":
    main()
