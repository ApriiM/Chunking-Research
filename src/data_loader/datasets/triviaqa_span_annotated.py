from __future__ import annotations

import hashlib
import heapq
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset

from src.data_loader.core.registry import dataset
from src.data_loader.core.schemas import DocumentRecord, QueryRecord


_ARTIFICIAL_TRAIN_SPLIT = "train_artificial"
_ARTIFICIAL_TEST_SPLIT = "test_artificial"
_ARTIFICIAL_QUERY_DOCS_PER_SPLIT = 10_000
_ARTIFICIAL_EXTRA_DOCS_PER_SPLIT = 20_000
_ARTIFICIAL_RANDOM_SEED = 42
_ARTIFICIAL_TOTAL_UNIQUE_DOCS = 2 * (
    _ARTIFICIAL_QUERY_DOCS_PER_SPLIT + _ARTIFICIAL_EXTRA_DOCS_PER_SPLIT
)


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


def _build_query_record(
    *,
    query_suffix: str,
    query_text: str,
    doc_id: str,
    spans: Any,
) -> QueryRecord:
    raw_spans, span_offsets = _normalize_spans(spans)
    return QueryRecord(
        query_id=f"q.triviaqa-span-annotated.{query_suffix}",
        contents=query_text,
        relevant=[doc_id],
        metadata={
            "dataset": "triviaqa-span-annotated",
            "spans": raw_spans,
            "span_offsets": span_offsets,
            "span_format": "char_offsets_[start,end)",
        },
    )


def _load_artificial_split(
    *,
    cache_dir: Optional[str],
    dataset_name: str,
    config_name: Optional[str],
    revision: Optional[str],
    show_progress: bool,
    requested_split: str,
    limit: Optional[int],
) -> Tuple[List[DocumentRecord], List[QueryRecord]]:
    """Build a deterministic synthetic split from the upstream test split.

    The upstream dataset only exposes a single `test` split. We derive:
      - train_artificial: 10k query-doc pairs + 20k extra docs
      - test_artificial: 10k disjoint query-doc pairs + 20k disjoint extra docs
    """
    load_kwargs: Dict[str, Any] = {
        "path": dataset_name,
        "split": "test",
        "cache_dir": cache_dir,
        "revision": revision,
        "streaming": True,
    }
    if config_name is not None:
        load_kwargs["name"] = config_name

    ds = load_dataset(**load_kwargs)

    row_iter = _progress_iter(
        ds,
        enabled=show_progress,
        desc="Selecting TriviaQA artificial split rows",
        total=None,
    )

    seen_doc_digests = set()
    kept_docs: Dict[str, Dict[str, Any]] = {}
    kept_heap: List[Tuple[int, str]] = []

    for idx, row in enumerate(row_iter):
        query_text = str(row.get("query", "") or "")
        document_text = str(row.get("document", "") or "")
        if not query_text or not document_text:
            continue

        doc_digest = hashlib.md5(document_text.encode("utf-8")).hexdigest()
        if doc_digest in seen_doc_digests:
            continue
        seen_doc_digests.add(doc_digest)

        score_digest = hashlib.md5(
            f"{_ARTIFICIAL_RANDOM_SEED}:{doc_digest}".encode("utf-8")
        ).hexdigest()
        score = int(score_digest, 16)
        heap_entry = (-score, doc_digest)

        query_suffix = (
            str(row.get("id"))
            if "id" in row and row.get("id") is not None
            else str(idx)
        )
        candidate = {
            "doc_digest": doc_digest,
            "doc_id": f"triviaqa-doc-{doc_digest[:16]}",
            "document_text": document_text,
            "query_suffix": query_suffix,
            "query_text": query_text,
            "spans": row.get("spans"),
            "score": score,
        }

        if len(kept_docs) < _ARTIFICIAL_TOTAL_UNIQUE_DOCS:
            heapq.heappush(kept_heap, heap_entry)
            kept_docs[doc_digest] = candidate
            continue

        if heap_entry <= kept_heap[0]:
            continue

        _, removed_digest = heapq.heapreplace(kept_heap, heap_entry)
        kept_docs.pop(removed_digest, None)
        kept_docs[doc_digest] = candidate

    if len(kept_docs) < _ARTIFICIAL_TOTAL_UNIQUE_DOCS:
        raise ValueError(
            "Not enough unique documents to build artificial TriviaQA splits. "
            f"Need at least {_ARTIFICIAL_TOTAL_UNIQUE_DOCS}, got {len(kept_docs)}."
        )

    selected = sorted(
        kept_docs.values(),
        key=lambda item: (item["score"], item["doc_digest"]),
    )

    train_query_end = _ARTIFICIAL_QUERY_DOCS_PER_SPLIT
    train_extra_end = train_query_end + _ARTIFICIAL_EXTRA_DOCS_PER_SPLIT
    test_query_end = train_extra_end + _ARTIFICIAL_QUERY_DOCS_PER_SPLIT
    test_extra_end = test_query_end + _ARTIFICIAL_EXTRA_DOCS_PER_SPLIT

    if requested_split == _ARTIFICIAL_TRAIN_SPLIT:
        query_candidates = selected[:train_query_end]
        extra_candidates = selected[train_query_end:train_extra_end]
    elif requested_split == _ARTIFICIAL_TEST_SPLIT:
        query_candidates = selected[train_extra_end:test_query_end]
        extra_candidates = selected[test_query_end:test_extra_end]
    else:
        raise ValueError(f"Unsupported artificial split: {requested_split}")

    documents = [
        DocumentRecord(
            doc_id=item["doc_id"],
            contents=item["document_text"],
            metadata={"dataset": "triviaqa-span-annotated"},
        )
        for item in (query_candidates + extra_candidates)
    ]

    queries = [
        _build_query_record(
            query_suffix=item["query_suffix"],
            query_text=item["query_text"],
            doc_id=item["doc_id"],
            spans=item["spans"],
        )
        for item in query_candidates
    ]

    if limit is not None:
        queries = queries[: min(limit, len(queries))]

    return documents, queries


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

    Additional synthetic splits:
      - train_artificial: 10k query-doc pairs + 20k extra docs
      - test_artificial: 10k disjoint query-doc pairs + 20k disjoint extra docs
    """
    base_split, split_slice = _split_base_and_slice(split)
    if base_split in (_ARTIFICIAL_TRAIN_SPLIT, _ARTIFICIAL_TEST_SPLIT):
        return _load_artificial_split(
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            config_name=config_name,
            revision=revision,
            show_progress=show_progress,
            requested_split=base_split,
            limit=limit,
        )

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

        if "id" in row and row.get("id") is not None:
            query_suffix = str(row.get("id"))
        else:
            query_suffix = str(idx)

        queries.append(
            _build_query_record(
                query_suffix=query_suffix,
                query_text=query_text,
                doc_id=doc_id,
                spans=row.get("spans"),
            )
        )

    return documents, queries
