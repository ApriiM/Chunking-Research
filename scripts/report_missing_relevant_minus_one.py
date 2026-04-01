#!/usr/bin/env python3
"""Report per-dataset count of queries with artificial relevant id '-1'."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results_converter.missing_relevant_report import (
    build_missing_relevant_report,
    render_report_tsv,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for missing relevant report generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze exported PIRB queries and report per dataset how many queries "
            "contain relevant id '-1'."
        )
    )
    parser.add_argument(
        "--input-root",
        default="export_to_pirb",
        help="Root directory containing exported PIRB run_* folders.",
    )
    parser.add_argument(
        "--output-path",
        default="",
        help="Optional path to save TSV report. If empty, only stdout is used.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and print the missing relevant id report."""

    args = parse_args()
    input_root = Path(args.input_root)
    rows = build_missing_relevant_report(input_root)
    report_tsv = render_report_tsv(rows)
    print(report_tsv, end="")

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_tsv, encoding="utf-8")


if __name__ == "__main__":
    main()
