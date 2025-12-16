from typing import Callable, Dict, Optional

from .types import QASample

DatasetLoader = Callable[..., list[QASample]]

_DATASET_REGISTRY: Dict[str, DatasetLoader] = {}


def register_dataset(name: str, loader: DatasetLoader):
    if name in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' already registered")
    _DATASET_REGISTRY[name] = loader


def get_dataset_loader(name: str) -> DatasetLoader:
    loader = _DATASET_REGISTRY.get(name)
    if not loader:
        raise ValueError(f"Unknown dataset '{name}'. Registered: {list(_DATASET_REGISTRY.keys())}")
    return loader


def list_datasets():
    return list(_DATASET_REGISTRY.keys())
