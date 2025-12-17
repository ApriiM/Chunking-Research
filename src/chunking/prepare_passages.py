import argparse
import os
from typing import Dict, List

import yaml

from src.chunking import get_chunker
from src.data_loader.core.schemas import (
    PassageRecord,
    load_document_records_jsonl,
    save_passage_records_jsonl,
)


def _parse_chunker_params(raw: str) -> Dict:
    if not raw:
        return {}
    parsed = yaml.safe_load(raw)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("chunker_params must decode to a mapping")
    return parsed


def chunk_documents(
    documents_path: str,
    chunker_name: str,
    chunker_params: Dict,
    output_path: str,
    overwrite: bool,
) -> None:
    documents = load_document_records_jsonl(documents_path)
    chunker = get_chunker(chunker_name, chunker_params)

    passages: List[PassageRecord] = []
    for doc in documents:
        base_meta = {"doc_id": doc.doc_id, **(doc.metadata or {})}
        for chunk in chunker.split_text(doc.contents, document_meta=base_meta):
            metadata = dict(chunk.metadata or {})
            if "doc_id" not in metadata:
                metadata["doc_id"] = doc.doc_id
            passages.append(
                PassageRecord(
                    passage_id=str(chunk.chunk_id),
                    contents=chunk.text,
                    parent_id=doc.doc_id,
                    metadata=metadata,
                )
            )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if (not overwrite) and os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    save_passage_records_jsonl(passages, output_path)
    print(f"Wrote {len(passages)} passages to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Chunk documents.jsonl into passages.jsonl.")
    parser.add_argument("--documents-path", required=True, help="Path to documents.jsonl")
    parser.add_argument("--chunker-name", required=True, help="Registered chunker name")
    parser.add_argument(
        "--chunker-params",
        default="",
        help="YAML/JSON string with params forwarded to the chunker",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to write passages.jsonl (defaults alongside documents.jsonl)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    chunker_params = _parse_chunker_params(args.chunker_params)
    output_path = args.output_path
    if output_path is None:
        base_dir = os.path.dirname(args.documents_path)
        output_path = os.path.join(base_dir, "passages.jsonl")

    chunk_documents(
        documents_path=args.documents_path,
        chunker_name=args.chunker_name,
        chunker_params=chunker_params,
        output_path=output_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
