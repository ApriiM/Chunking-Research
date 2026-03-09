#!/usr/bin/env python3
"""Export experiment run outputs into PIRB-oriented folder structure."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.results_converter.pirb_export import export_runs_to_pirb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export run folders (session or single run) into export_to_pirb structure, "
            "copying metadata.json, queries/queries.jsonl and passages/passages.jsonl."
        )
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help=(
            "Path to experiment session folder (containing run_* subfolders) "
            "or directly to a single run_* folder."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="export_to_pirb",
        help="Destination root folder for exported runs (default: export_to_pirb)",
    )
    parser.add_argument(
        "--overwrite-run-dir",
        action="store_true",
        help="If target run directory exists, remove and recreate it before copy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    output_root = Path(args.output_root)

    summary = export_runs_to_pirb(
        input_path=input_path,
        output_root=output_root,
        overwrite_run_dir=bool(args.overwrite_run_dir),
        repo_root=Path.cwd(),
        log_fn=print,
    )

    success_count = len(summary.successes)
    failed_count = len(summary.failures)
    print(f"\nExport destination: {summary.output_root}")

    for result in summary.successes:
        print(f"- {result.source_run_dir}")
        print(f"  -> {result.target_run_dir}")

    if summary.failures:
        print("\nFailed run(s):")
        for failure in summary.failures:
            print(f"- {failure.source_run_dir}")
            print(f"  reason: {failure.reason}")

    reason_counter = Counter(f.reason for f in summary.failures)
    print("\nSummary:")
    print(f"- successfully converted: {success_count}")
    print(f"- failed: {failed_count}")
    if reason_counter:
        print("- failure reasons:")
        for reason, count in reason_counter.items():
            print(f"  - {count}x {reason}")


if __name__ == "__main__":
    main()
