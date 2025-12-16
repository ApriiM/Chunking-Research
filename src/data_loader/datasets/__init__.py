"""Dataset-specific loaders live here and self-register via the registry."""

from src.data_loader.registry import register_dataset
from .poquad import load_poquad

# register built-in datasets
register_dataset("poquad", load_poquad)

__all__ = ["load_poquad"]
