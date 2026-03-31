#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def find_document_files(processed_root: Path) -> List[Path]:
    return sorted(processed_root.glob("**/documents/documents.jsonl"))


def dataset_slug_from_doc_path(doc_path: Path, processed_root: Path) -> str:
    dataset_root = doc_path.parent.parent
    return str(dataset_root.relative_to(processed_root))


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}") from exc


def analyze_dataset(doc_path: Path, threshold: int) -> Tuple[int, int, float, int]:
    total = 0
    over = 0
    max_len = 0

    for row in iter_jsonl(doc_path):
        text = row.get("contents", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        length = len(text)
        total += 1
        if length > max_len:
            max_len = length
        if length > threshold:
            over += 1

    pct = (over / total * 100.0) if total else 0.0
    return total, over, pct, max_len


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report per-dataset document lengths from data/processed/**/documents/documents.jsonl"
        )
    )
    parser.add_argument(
        "--processed-root",
        default="data/processed",
        help="Root directory to scan for datasets (default: data/processed)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1_000_000,
        help="Character threshold for counting long docs (default: 1000000)",
    )
    args = parser.parse_args()

    processed_root = Path(args.processed_root).resolve()
    doc_files = find_document_files(processed_root)

    if not doc_files:
        print(f"No documents found under: {processed_root}")
        return

    print(f"Processed root: {processed_root}")
    print(f"Threshold: {args.threshold} chars")
    print(f"Datasets found: {len(doc_files)}")
    print("")
    print(
        "dataset\ttotal_docs\tdocs_over_threshold\tpercent_over_threshold\tlongest_doc_chars"
    )

    for doc_path in doc_files:
        dataset_slug = dataset_slug_from_doc_path(doc_path, processed_root)
        total, over, pct, max_len = analyze_dataset(doc_path, args.threshold)
        print(f"{dataset_slug}\t{total}\t{over}\t{pct:.2f}\t{max_len}")


if __name__ == "__main__":
    main()
