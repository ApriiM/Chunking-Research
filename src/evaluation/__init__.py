from typing import Iterable, List, Tuple

from .dummy_metrics import calculate_chunk_stats
from .retrieval_eval import retrieval_at_k
from .registry import get_evaluation, register_evaluation, resolve_evaluations

# Register built-ins
register_evaluation("chunk_stats", calculate_chunk_stats)
register_evaluation("retrieval_at_k", retrieval_at_k)


def get_evaluations(names: Iterable[str]) -> List[Tuple[str, callable]]:
    return resolve_evaluations(names)


__all__ = ["get_evaluations", "register_evaluation", "get_evaluation", "resolve_evaluations"]
