import os
from pathlib import Path
from typing import Any, Dict


_DEFAULT_RELATIVE_CACHE_DIR = "data/hf_cached_models"


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on", "y"}:
            return True
        if lowered in {"false", "0", "no", "off", "n"}:
            return False
    return default


def _repo_root() -> Path:
    # .../src/chunking/core/hf_cache.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def resolve_hf_cache_dir(config: Dict[str, Any] | None = None) -> str:
    cfg = config or {}
    raw = cfg.get("hf_cache_dir") or os.environ.get("HF_CACHE_DIR") or _DEFAULT_RELATIVE_CACHE_DIR

    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = _repo_root() / p

    cache_dir = p.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "hub").mkdir(parents=True, exist_ok=True)

    # Keep all Hugging Face-related caches in one place unless user overrode them.
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir))

    return str(cache_dir)


def hf_local_files_only(config: Dict[str, Any] | None = None) -> bool:
    cfg = config or {}
    return _coerce_bool(cfg.get("hf_local_files_only"), default=False)

