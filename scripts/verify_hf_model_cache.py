#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer


HF_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "BAAI/bge-m3",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def _configure_cache_env(cache_dir: Path) -> None:
    hub_dir = cache_dir / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_dir)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir)


def _snapshot(model_id: str, cache_dir: Path, local_only: bool) -> Path:
    path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        local_files_only=local_only,
        resume_download=True,
    )
    return Path(path)


def _verify_qwen_shards(snapshot_dir: Path) -> None:
    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    if not weight_map:
        raise RuntimeError(f"No weight_map in {index_path}")

    shard_names = sorted(set(weight_map.values()))
    missing = [name for name in shard_names if not (snapshot_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing shard files in Qwen snapshot: " + ", ".join(missing)
        )


def _prefetch(models: Iterable[str], cache_dir: Path) -> None:
    for model_id in models:
        print(f"[prefetch] {model_id}")
        snapshot_path = _snapshot(model_id, cache_dir=cache_dir, local_only=False)
        print(f"  -> {snapshot_path}")


def _verify_offline(models: Iterable[str], cache_dir: Path) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    for model_id in models:
        print(f"[verify-offline] {model_id}")
        snapshot_path = _snapshot(model_id, cache_dir=cache_dir, local_only=True)
        print(f"  snapshot: {snapshot_path}")

        # Tokenizer + config verification catches many partial-cache states.
        AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        AutoConfig.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )

        if model_id == "Qwen/Qwen3-4B-Instruct-2507":
            _verify_qwen_shards(snapshot_path)
            print("  qwen shards: OK")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prefetch and verify HF model cache for chunking methods."
    )
    parser.add_argument(
        "--cache-dir",
        default="data/hf_cached_models",
        help="Root directory for HF cache (default: data/hf_cached_models).",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Download/fill cache from HF Hub.",
    )
    parser.add_argument(
        "--verify-offline",
        action="store_true",
        help="Validate that required files are present with offline-only mode.",
    )
    args = parser.parse_args()

    if not args.prefetch and not args.verify_offline:
        args.prefetch = True
        args.verify_offline = True

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    _configure_cache_env(cache_dir)

    print(f"Cache dir: {cache_dir}")
    print("Models:")
    for mid in HF_MODELS:
        print(f"  - {mid}")

    try:
        if args.prefetch:
            _prefetch(HF_MODELS, cache_dir)
        if args.verify_offline:
            _verify_offline(HF_MODELS, cache_dir)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
