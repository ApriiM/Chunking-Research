import os
import yaml
from copy import deepcopy
from typing import Dict


def _chunker_defaults_path(name: str) -> str:
    """Return the path to the chunker-specific defaults YAML."""

    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "chunkers", f"{name}.yaml")
    )


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
