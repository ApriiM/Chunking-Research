from typing import Any, Iterable, TypeVar


T = TypeVar("T")


def coerce_progress_enabled(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def iter_with_progress(
    items: Iterable[T],
    *,
    desc: str,
    enabled: bool = True,
) -> Iterable[T]:
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        return items

    total = len(items) if hasattr(items, "__len__") else None
    return tqdm(items, total=total, desc=desc, unit="doc")
