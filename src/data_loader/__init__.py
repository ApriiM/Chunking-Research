"""Dataset loaders and helpers grouped by role."""

from .core.registry import get_dataset_loader, list_datasets, register_dataset

# import datasets package to trigger auto-registration
from . import datasets  # noqa: F401
