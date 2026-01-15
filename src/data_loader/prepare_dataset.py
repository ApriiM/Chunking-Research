import argparse
import os
from typing import Dict, Optional

import yaml

from src.data_loader.core.registry import get_dataset_loader
from src.data_loader.core.schemas import save_document_records_jsonl, save_query_records_jsonl


def _parse_loader_kwargs(raw: str) -> Dict:
    if not raw:
        return {}
    parsed = yaml.safe_load(raw)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("loader_kwargs must decode to a mapping")
    return parsed

def prepare_dataset(
    dataset: str,
    split: str,
    output_dir: str,
    cache_dir: Optional[str],
    limit: Optional[int],
    loader_kwargs: Dict,
    overwrite: bool,
) -> None:

    loader = get_dataset_loader(dataset)
    documents, queries = loader(split=split, cache_dir=cache_dir, limit=limit, **loader_kwargs)

    documents_dir = os.path.join(output_dir, "documents")
    queries_dir = os.path.join(output_dir, "queries")
    passages_dir = os.path.join(output_dir, "passages")
    passages_all_dir = os.path.join(output_dir, "passages_all")
    for path in (documents_dir, queries_dir, passages_dir, passages_all_dir):
        os.makedirs(path, exist_ok=True)
    documents_path = os.path.join(documents_dir, "documents.jsonl")
    queries_path = os.path.join(queries_dir, "queries.jsonl")

    if not overwrite:
        for path in (documents_path, queries_path):
            if os.path.exists(path):
                raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    save_document_records_jsonl(documents, documents_path)
    save_query_records_jsonl(queries, queries_path)
    print(f"Wrote {len(documents)} documents to {documents_path}")
    print(f"Wrote {len(queries)} queries to {queries_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset into documents/queries JSONL.")
    parser.add_argument("--dataset", required=True, help="Registered dataset loader name")
    parser.add_argument("--split", default="train", help="Dataset split or slice expression")
    parser.add_argument("--output-dir", default=None, help="Output directory (defaults to data/processed/<dataset>/<split>)")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory")
    parser.add_argument("--limit", type=int, default=None, help="Optional hard cap on number of rows")
    parser.add_argument(
        "--loader-kwargs",
        default="",
        help="YAML/JSON string with extra kwargs forwarded to the dataset loader",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    loader_kwargs = _parse_loader_kwargs(args.loader_kwargs)
    output_dir = args.output_dir or os.path.join("data", "processed", args.dataset, args.split)

    prepare_dataset(
        dataset=args.dataset,
        split=args.split,
        output_dir=output_dir,
        cache_dir=args.cache_dir,
        limit=args.limit,
        loader_kwargs=loader_kwargs,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
