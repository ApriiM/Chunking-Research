from __future__ import annotations

import hashlib
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset

from src.data_loader.core.registry import dataset
from src.data_loader.core.schemas import DocumentRecord, QueryRecord


def _parse_span(span_value: Any) -> Optional[Tuple[int, int]]:
    """Parse a span value into (start, end) character offsets when possible."""
    if isinstance(span_value, str):
        left, sep, right = span_value.partition(":")
        if sep != ":":
            return None
        try:
            start = int(left.strip())
            end = int(right.strip())
        except ValueError:
            return None
        if start < 0 or end <= start:
            return None
        return (start, end)

    if isinstance(span_value, (list, tuple)) and len(span_value) == 2:
        try:
            start = int(span_value[0])
            end = int(span_value[1])
        except (TypeError, ValueError):
            return None
        if start < 0 or end <= start:
            return None
        return (start, end)

    return None


def _normalize_spans(raw_spans: Any) -> Tuple[List[str], List[List[int]]]:
    """Return (raw span strings, parsed offsets [[start, end], ...])."""
    if not isinstance(raw_spans, list):
        return ([], [])

    raw_as_strings: List[str] = []
    parsed_offsets: List[List[int]] = []

    for span_value in raw_spans:
        raw_as_strings.append(str(span_value))
        parsed = _parse_span(span_value)
        if parsed is not None:
            parsed_offsets.append([parsed[0], parsed[1]])

    return (raw_as_strings, parsed_offsets)


def _split_base_and_slice(split_expr: str) -> Tuple[str, Optional[slice]]:
    """Parse expressions like 'test[:100]' into ('test', slice(None, 100))."""
    if not split_expr or "[" not in split_expr or not split_expr.endswith("]"):
        return split_expr, None

    base, bracket = split_expr.split("[", 1)
    base = base.strip()
    inner = bracket[:-1].strip()

    if ":" not in inner:
        return split_expr, None

    left, right = inner.split(":", 1)
    left = left.strip()
    right = right.strip()

    try:
        start = int(left) if left else None
        stop = int(right) if right else None
    except ValueError:
        return split_expr, None
    return (base, slice(start, stop))


def _progress_iter(
    items: Iterable[Any],
    *,
    enabled: bool,
    desc: str,
    total: Optional[int] = None,
):
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except Exception:
        return items
    return tqdm(items, desc=desc, total=total)


@dataset("triviaqa_span_annotated")
@dataset("triviaqa-span-annotated")
def load_triviaqa_span_annotated(
    split: str = "test",
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    dataset_name: str = "jinaai/triviaqa-span-annotated",
    config_name: Optional[str] = "default",
    revision: Optional[str] = None,
    streaming: bool = False,
    show_progress: bool = True,
) -> Tuple[List[DocumentRecord], List[QueryRecord]]:
    """
    Load TriviaQA span-annotated rows as normalized documents + queries.

    Query metadata includes:
      - spans: original span values from the dataset
      - span_offsets: parsed character offsets as [[start, end], ...]
      - span_format: fixed descriptor for offset semantics
    """
    base_split, split_slice = _split_base_and_slice(split)
    auto_streaming = (
        split_slice is not None
        and (split_slice.start is None or split_slice.start == 0)
        and split_slice.stop is not None
    )
    effective_streaming = bool(streaming or auto_streaming)

    load_kwargs: Dict[str, Any] = {
        "path": dataset_name,
        "split": base_split if effective_streaming else split,
        "cache_dir": cache_dir,
        "revision": revision,
    }
    if config_name is not None:
        load_kwargs["name"] = config_name
    if effective_streaming:
        load_kwargs["streaming"] = True

    ds = load_dataset(**load_kwargs)
    if not effective_streaming and limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    documents: List[DocumentRecord] = []
    queries: List[QueryRecord] = []

    document_to_id: Dict[str, str] = {}

    if effective_streaming:
        start = split_slice.start if split_slice and split_slice.start is not None else 0
        stop = split_slice.stop if split_slice else None

        if stop is not None:
            max_rows = max(0, stop - start)
            if limit is not None:
                max_rows = min(max_rows, max(0, int(limit)))
        else:
            max_rows = max(0, int(limit)) if limit is not None else None

        row_iter = islice(ds, start, None)
        if max_rows is not None:
            row_iter = islice(row_iter, max_rows)
        progress_total: Optional[int] = max_rows
    else:
        row_iter = ds
        progress_total = len(ds)

    row_iter = _progress_iter(
        row_iter,
        enabled=show_progress,
        desc="Loading TriviaQA rows",
        total=progress_total,
    )

    for idx, row in enumerate(row_iter):
        query_text = str(row.get("query", "") or "")
        document_text = str(row.get("document", "") or "")
        if not query_text or not document_text:
            continue

        doc_id = document_to_id.get(document_text)
        if doc_id is None:
            digest = hashlib.md5(document_text.encode("utf-8")).hexdigest()[:16]
            doc_id = f"triviaqa-doc-{digest}"
            document_to_id[document_text] = doc_id
            documents.append(
                DocumentRecord(
                    doc_id=doc_id,
                    contents=document_text,
                    metadata={"dataset": "triviaqa-span-annotated"},
                )
            )

        raw_spans, span_offsets = _normalize_spans(row.get("spans"))
        metadata: Dict[str, Any] = {
            "dataset": "triviaqa-span-annotated",
            "spans": raw_spans,
            "span_offsets": span_offsets,
            "span_format": "char_offsets_[start,end)",
        }

        if "id" in row and row.get("id") is not None:
            query_suffix = str(row.get("id"))
        else:
            query_suffix = str(idx)

        queries.append(
            QueryRecord(
                query_id=f"q.triviaqa-span-annotated.{query_suffix}",
                contents=query_text,
                relevant=[doc_id],
                metadata=metadata,
            )
        )

    return documents, queries
