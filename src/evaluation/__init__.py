from typing import Iterable, List, Tuple

from .core import evaluation, get_evaluation, list_evaluations, register_evaluation, resolve_evaluations
from . import metrics  # noqa: F401 triggers auto-registration via decorators


def get_evaluations(names: Iterable[str]) -> List[Tuple[str, callable]]:
    return resolve_evaluations(names)


__all__ = [
    "evaluation",
    "get_evaluation",
    "get_evaluations",
    "list_evaluations",
    "register_evaluation",
    "resolve_evaluations",
]
