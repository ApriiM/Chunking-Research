from typing import Callable, Iterable, List, Tuple

from .dummy_metrics import calculate_chunk_stats

# Registry of available evaluation functions
EVALUATIONS = {
    "chunk_stats": calculate_chunk_stats,
}


def get_evaluations(names: Iterable[str]) -> List[Tuple[str, Callable]]:
    """Resolve evaluation names to callables.

    :param names: Iterable of evaluation names or a single name string
    :raises ValueError: if an unknown evaluation name is provided
    :return: List of (name, callable) pairs in requested order
    """

    if names is None:
        return []

    if isinstance(names, str):
        names = [names]

    resolved: List[Tuple[str, Callable]] = []
    for name in names:
        fn = EVALUATIONS.get(name)
        if not fn:
            raise ValueError(f"Unknown evaluation '{name}'. Available: {list(EVALUATIONS.keys())}")
        resolved.append((name, fn))
    return resolved
