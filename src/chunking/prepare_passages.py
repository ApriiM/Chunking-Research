import argparse
import json
import os
from datetime import datetime, timezone
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
    document_texts = [doc.contents for doc in documents]
    documents_meta = [{"doc_id": doc.doc_id, **(doc.metadata or {})} for doc in documents]

    for chunk in chunker.split_text(document_texts, documents_meta=documents_meta):
        metadata = dict(chunk.metadata or {})
        doc_id = metadata.get("doc_id")
        passages.append(
            PassageRecord(
                passage_id=str(chunk.chunk_id),
                contents=chunk.text,
                parent_id=str(doc_id) if doc_id is not None else "",
                metadata=metadata,
            )
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    meta_path = _metadata_path(output_path)

    if not overwrite:
        if os.path.exists(output_path):
            raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
        if os.path.exists(meta_path):
            raise FileExistsError(f"Refusing to overwrite existing metadata file: {meta_path}")

    save_passage_records_jsonl(passages, output_path)
    _write_metadata(
        meta_path,
        {
            "documents_path": documents_path,
            "output_path": output_path,
            "chunker_name": chunker_name,
            "chunker_params": chunker_params,
            "overwrite": overwrite,
            "document_count": len(documents),
            "chunk_count": len(passages),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Wrote {len(passages)} passages to {output_path}")
    print(f"Wrote metadata to {meta_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Chunk documents.jsonl into passages.jsonl.")
    parser.add_argument("--config", default=None, help="Path to experiment YAML; overrides other flags")
    parser.add_argument("--documents-path", help="Path to documents.jsonl")
    parser.add_argument("--chunker-name", help="Registered chunker name")
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


def _load_experiment_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must decode to a mapping")
    if "documents_path" not in data:
        raise ValueError("Config must include documents_path")
    chunker_cfg = data.get("chunker") or {}
    if not isinstance(chunker_cfg, dict) or "name" not in chunker_cfg:
        raise ValueError("Config must include chunker.name")
    params = chunker_cfg.get("params", {}) or {}
    if not isinstance(params, dict):
        raise ValueError("chunker.params must be a mapping if provided")
    output_cfg = data.get("output", {}) or {}
    if output_cfg and not isinstance(output_cfg, dict):
        raise ValueError("output section must be a mapping if provided")
    return {
        "documents_path": data["documents_path"],
        "chunker_name": chunker_cfg["name"],
        "chunker_params": params,
        "output_path": data.get("output_path"),
        "overwrite": _coerce_bool(output_cfg.get("overwrite", data.get("overwrite", False))),
    }


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if value is None:
        return False
    raise ValueError(f"overwrite must be a boolean (got {value!r})")


def _metadata_path(output_path: str) -> str:
    base, _ = os.path.splitext(output_path)
    return f"{base}.meta.json"


def _write_metadata(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    if args.config:
        cfg = _load_experiment_config(args.config)
        documents_path = cfg["documents_path"]
        chunker_name = cfg["chunker_name"]
        chunker_params = cfg["chunker_params"]
        output_path = cfg["output_path"]
        overwrite = cfg["overwrite"]
    else:
        if not args.documents_path or not args.chunker_name:
            raise ValueError("Either --config or both --documents-path and --chunker-name are required")
        documents_path = args.documents_path
        chunker_name = args.chunker_name
        chunker_params = _parse_chunker_params(args.chunker_params)
        output_path = args.output_path
        overwrite = args.overwrite

    if output_path is None:
        base_dir = os.path.dirname(documents_path)
        output_path = os.path.join(base_dir, "passages.jsonl")

    chunk_documents(
        documents_path=documents_path,
        chunker_name=chunker_name,
        chunker_params=chunker_params,
        output_path=output_path,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()
