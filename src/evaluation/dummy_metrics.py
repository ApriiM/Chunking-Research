"""
Example dummy metrics for evaluating chunking performance.
"""

from typing import List, Dict
from src.chunking.base import Chunk
import numpy as np

def calculate_chunk_stats(chunks: List[Chunk]) -> Dict[str, float]:
    """
    Calculates basic statistics about the generated chunks.
    """
    lengths = [len(c.text) for c in chunks]
    
    if not lengths:
        return {
            "chunk_count": 0,
            "avg_length": 0.0,
            "min_length": 0,
            "max_length": 0,
            "std_dev_length": 0.0,
        }

    stats = {
        "chunk_count": len(chunks),
        "avg_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "std_dev_length": float(np.std(lengths))
    }
    return stats