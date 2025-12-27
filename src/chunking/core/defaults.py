import os
from pathlib import Path
import yaml
from copy import deepcopy
from typing import Dict


def _chunker_defaults_path(name: str) -> str:
    """Resolve the chunker defaults YAML path by walking up to the repo root."""

    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        candidate = cand / "configs" / "chunkers" / f"{name}.yaml"
        if candidate.exists():
            return str(candidate)
    # Fall back to the expected repo layout even if the file does not exist
    return str(here.parents[2] / "configs" / "chunkers" / f"{name}.yaml")

_DEFAULT_CACHE: Dict[str, Dict[str, object]] = {}


def _load_defaults(name: str) -> Dict[str, object]:
    path = _chunker_defaults_path(name)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Chunker defaults file must be a mapping: {path}")
    return data


def merge_with_defaults(name: str, overrides: Dict[str, object]) -> Dict[str, object]:
    """Merge overrides with chunker-specific defaults from configs/chunkers/{name}.yaml."""

    if name not in _DEFAULT_CACHE:
        _DEFAULT_CACHE[name] = _load_defaults(name)
    base = deepcopy(_DEFAULT_CACHE.get(name, {}) or {})
    if overrides:
        base.update(overrides)
    return base
