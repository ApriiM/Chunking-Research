import importlib
from typing import Dict, List, Type

from .base import BaseChunker
from .defaults import merge_with_defaults

ChunkerFactory = Type[BaseChunker]
_CHUNKER_REGISTRY: Dict[str, ChunkerFactory] = {}
_BUILTIN_CHUNKERS_LOADED = False


def _ensure_builtin_chunkers_loaded() -> None:
    global _BUILTIN_CHUNKERS_LOADED

    if _BUILTIN_CHUNKERS_LOADED:
        return

    importlib.import_module("src.chunking.strategies")
    _BUILTIN_CHUNKERS_LOADED = True


def register_chunker(name: str, chunker_cls: ChunkerFactory) -> None:
    if name in _CHUNKER_REGISTRY:
        raise ValueError(f"Chunker '{name}' already registered")
    _CHUNKER_REGISTRY[name] = chunker_cls


def chunker(name: str):
    """Decorator to register a chunker class under a given name."""

    def decorator(cls: ChunkerFactory) -> ChunkerFactory:
        register_chunker(name, cls)
        return cls

    return decorator


def list_chunkers() -> List[str]:
    _ensure_builtin_chunkers_loaded()
    return list(_CHUNKER_REGISTRY.keys())


def get_chunker(name: str, config: Dict) -> BaseChunker:
    _ensure_builtin_chunkers_loaded()
    if name not in _CHUNKER_REGISTRY:
        raise ValueError(f"Chunker '{name}' not found. Available: {list_chunkers()}")
    chunker_cls = _CHUNKER_REGISTRY[name]
    merged_config = merge_with_defaults(name, config or {})
    return chunker_cls(merged_config)
