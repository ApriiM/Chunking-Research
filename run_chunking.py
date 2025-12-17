import argparse
import json
import os
import sys
import yaml
from datetime import datetime


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.chunking import get_chunker
from src.data_loader import load_text_file


def parse_args():
    '''
    Parses command line arguments.
    '''
    parser = argparse.ArgumentParser(description="Run a chunking experiment.")
    parser.add_argument(
        "--config",
        default="configs/experiments/chunking/fixed_size_demo.yaml",
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
    Load input text data based on the config.
    
    :param config: Full experiment configuration
    :type config: dict
    '''
    if "input_file" in config:
        input_path = config["input_file"]
        print(f"Loading data from file: {input_path}")
        text_data = load_text_file(input_path)
        doc_meta = {"source": input_path}
    elif "input_text" in config:
        text_data = config["input_text"]
        doc_meta = {"source": "inline_text"}
    else:
        raise ValueError("Config must include either input_file or input_text")
    return text_data, doc_meta


def load_output_config(config: dict):
    '''Extract output settings such as saving chunks to disk.'''
    output_cfg = config.get("output") or {}
    save_chunks = bool(output_cfg.get("save_chunks", False))
    chunks_path = output_cfg.get("chunks_path", "results/chunks.jsonl")
    overwrite = bool(output_cfg.get("overwrite", False))
    return save_chunks, chunks_path, overwrite


def ensure_output_path(path: str, overwrite: bool) -> str:
    """Return a writable path; add timestamp suffix if not overwriting and file exists."""
    if overwrite or not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{base}_{ts}{ext}"


def save_chunks_to_file(chunks, output_path: str):
    '''Persist chunks as JSONL with text, id, and metadata.'''
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            payload = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks to {output_path}")

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

    # Load input text data and metadata
    text_data, doc_meta = load_chunker_input(cfg)

    # Initialize chunker
    print(f"Initializing '{chunker_name}' chunker with params: {chunker_params}")
    chunker = get_chunker(chunker_name, chunker_params)

    # Perform chunking
    print("Performing chunking...")
    chunks = chunker.split_text(text_data, document_meta=doc_meta)

    if save_chunks:
        final_path = ensure_output_path(chunks_path, overwrite)
        save_chunks_to_file(chunks, final_path)

if __name__ == "__main__":
    main()