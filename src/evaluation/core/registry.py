from typing import Callable, Dict, Iterable, List, Tuple

EvaluationFn = Callable[..., dict]

_EVAL_REGISTRY: Dict[str, EvaluationFn] = {}


def register_evaluation(name: str, fn: EvaluationFn):
    if name in _EVAL_REGISTRY:
        raise ValueError(f"Evaluation '{name}' already registered")
    _EVAL_REGISTRY[name] = fn


def evaluation(name: str):
    """Decorator to register an evaluation function by name."""

    def decorator(fn: EvaluationFn) -> EvaluationFn:
        register_evaluation(name, fn)
        return fn

    return decorator


def get_evaluation(name: str) -> EvaluationFn:
    fn = _EVAL_REGISTRY.get(name)
    if not fn:
        raise ValueError(f"Unknown evaluation '{name}'. Registered: {list(_EVAL_REGISTRY.keys())}")
    return fn


def list_evaluations() -> List[str]:
    return list(_EVAL_REGISTRY.keys())


def resolve_evaluations(requested: Iterable[str]) -> List[Tuple[str, EvaluationFn]]:
    if requested is None:
        return []
    if isinstance(requested, str):
        requested = [requested]
    resolved: List[Tuple[str, EvaluationFn]] = []
    for name in requested:
        resolved.append((name, get_evaluation(name)))
    return resolved
