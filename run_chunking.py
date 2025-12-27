import argparse
import json
import os
import sys
import yaml
from datetime import datetime, timezone


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.chunking import get_chunker
from src.data_loader.core.schemas import (
    ChunkRecord,
    save_chunk_records_jsonl,
    load_document_records_jsonl,
)


def parse_args():
    '''
    Parses command line arguments.
    '''
    parser = argparse.ArgumentParser(description="Run a chunking experiment.")
    parser.add_argument(
        "--config",
        default="configs/experiments/run_chunking_fixed_size.yaml",
        help="Path to experiment config YAML",
    )
    return parser.parse_args()


def load_config(config_path: str):
    '''
    Load YAML config file.
    
    :param config_path: Path to the config YAML file
    :type config_path: str
    '''
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_chunker_config(config: dict):
    '''
    Extract chunker configuration from the main config.
    
    :param config: Full experiment configuration
    :type config: dict
    '''
    chunker_cfg = config.get("chunker") or {}
    chunker_name = chunker_cfg.get("name")
    chunker_params = chunker_cfg.get("params", {})
    if not chunker_name:
        raise ValueError("Config must include chunker.name")
    return chunker_name, chunker_params

def load_chunker_input(config: dict):
    '''
    Load input documents from a JSONL file of DocumentRecord entries.
    
    :param config: Full experiment configuration
    :type config: dict
    '''
    documents_path = config.get("documents_path")
    if not documents_path:
        raise ValueError("Config must include documents_path (JSONL with DocumentRecord entries)")
    if not os.path.exists(documents_path):
        raise FileNotFoundError(f"File not found: {documents_path}")

    print(f"Loading documents from {documents_path}")
    records = load_document_records_jsonl(documents_path)
    documents = [rec.contents for rec in records]
    documents_meta = [
        {"doc_id": rec.doc_id, **(rec.metadata or {}), "source": documents_path}
        for rec in records
    ]
    return documents, documents_meta


def load_output_config(config: dict):
    '''Extract output settings such as saving chunks to disk.'''
    output_cfg = config.get("output") or {}
    save_chunks = bool(output_cfg.get("save_chunks", False))
    chunks_path = output_cfg.get("chunks_path", "results/chunks.jsonl")
    overwrite = _coerce_bool(output_cfg.get("overwrite", False), "output.overwrite")
    return save_chunks, chunks_path, overwrite


def ensure_output_path(path: str, overwrite: bool) -> str:
    """Return target path; when overwrite is False, append timestamp suffix always."""
    if overwrite:
        return path
    base, ext = os.path.splitext(path)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{base}_{ts}{ext}"


def _metadata_path(output_path: str) -> str:
    base, _ = os.path.splitext(output_path)
    return f"{base}.meta.json"


def _write_metadata(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _coerce_bool(value, field_name: str) -> bool:
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
    raise ValueError(f"{field_name} must be a boolean (got {value!r})")


def main():
    # Parse command line arguments
    args = parse_args()
    config_path = args.config

    # Load experiment config
    print(f"Loading config from {config_path}...")
    cfg = load_config(config_path)

    # Load chunker configuration
    chunker_name, chunker_params = load_chunker_config(cfg)

    # Output configuration
    save_chunks, chunks_path, overwrite = load_output_config(cfg)

    # Load input documents and metadata
    documents, documents_meta = load_chunker_input(cfg)

    # Initialize chunker
    print(f"Initializing '{chunker_name}' chunker with params: {chunker_params}")
    chunker = get_chunker(chunker_name, chunker_params)

    # Perform chunking
    print("Performing chunking...")
    chunks = chunker.split_text(documents, documents_meta=documents_meta)

    if save_chunks:
        final_path = ensure_output_path(chunks_path, overwrite)
        meta_path = _metadata_path(final_path)

        records = [ChunkRecord.from_chunk(chunk) for chunk in chunks]
        save_chunk_records_jsonl(records, final_path)
        _write_metadata(
            meta_path,
            {
                "config_path": config_path,
                "documents_path": cfg.get("documents_path"),
                "chunker_name": chunker_name,
                "chunker_params": chunker_params,
                "output_path": final_path,
                "overwrite": overwrite,
                "document_count": len(documents),
                "chunk_count": len(records),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"Saved {len(records)} chunks to {final_path}")
        print(f"Saved metadata to {meta_path}")

if __name__ == "__main__":
    main()