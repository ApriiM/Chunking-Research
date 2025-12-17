"""Dataset loaders and helpers grouped by role."""

from .core.registry import get_dataset_loader, list_datasets, register_dataset
from .core.types import QASample
from .io.jsonl import load_samples_jsonl, save_samples_jsonl
from .io.text import load_text_file

# import datasets package to trigger auto-registration
from . import datasets  # noqa: F401
