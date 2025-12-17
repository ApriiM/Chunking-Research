"""Chunk-level stats metrics."""

from typing import Dict, List

import numpy as np

from src.chunking.core.base import Chunk
from src.evaluation.core.registry import evaluation


@evaluation("chunk_stats")
def calculate_chunk_stats(chunks: List[Chunk]) -> Dict[str, float]:
    """Basic length statistics for chunks."""

    lengths = [len(c.text) for c in chunks]

    if not lengths:
        return {
            "chunk_count": 0,
            "avg_length": 0.0,
            "min_length": 0,
            "max_length": 0,
            "std_dev_length": 0.0,
        }

    return {
        "chunk_count": len(chunks),
        "avg_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "std_dev_length": float(np.std(lengths)),
    }
