from __future__ import annotations

import ast
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

from src.data_loader.core.registry import dataset
from src.data_loader.core.schemas import DocumentRecord, QueryRecord


_UNKNOWN_SPLIT_RE = re.compile(r"Should be one of (\[.*\])\.", re.DOTALL)


def _parse_available_splits(err_msg: str) -> Optional[List[str]]:
    match = _UNKNOWN_SPLIT_RE.search(err_msg)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(1))
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return value
    except Exception:
        return None
    return None


def _split_base_and_slice(split_expr: str) -> Tuple[str, Optional[slice]]:
    if "[" not in split_expr or not split_expr.endswith("]"):
        return split_expr, None

    base, bracket = split_expr.split("[", 1)
    base = base.strip()
    inner = bracket[:-1].strip()

    if ":" not in inner:
        return base, None

    left, right = inner.split(":", 1)
    left = left.strip()
    right = right.strip()

    start = int(left) if left else None
    stop = int(right) if right else None
    return base, slice(start, stop)


def _load_with_revision_fallback(
    *,
    path: str,
    name: str,
    split: str,
    cache_dir: Optional[str],
    revision: Optional[str],
):
    base, sl = _split_base_and_slice(split)
    slice_suffix = ""
    if sl is not None:
        a = "" if sl.start is None else str(sl.start)
        b = "" if sl.stop is None else str(sl.stop)
        slice_suffix = f"[{a}:{b}]"

    def _attempt(rev: Optional[str], split_expr: str):
        return load_dataset(
            path=path,
            name=name,
            split=split_expr,
            cache_dir=cache_dir,
            revision=rev,
        )

    try:
        return _attempt(revision, split)
    except ValueError as exc:
        msg = str(exc)
        if "BuilderConfig" in msg and "not found" in msg:
            return _attempt(None, split)
        if "Unknown split" in msg:
            avail = _parse_available_splits(msg)
            if avail and len(avail) == 1:
                fallback_split = f"{avail[0]}{slice_suffix}"
                return _attempt(revision, fallback_split)
        raise


@dataset("nfcorpus")
@dataset("NFCorpus")
def load_nfcorpus(
    split: str = "test",
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    dataset_name: str = "mteb/nfcorpus",
    revision: Optional[str] = None,
    corpus_subset: str = "corpus",
    queries_subset: str = "queries",
    qrels_subset: str = "default",
    doc_id_prefix: str = "nfcorpus-",
    min_rel_score: float = 1.0,
) -> Tuple[List[DocumentRecord], List[QueryRecord]]:
    qrels_base, qrels_slice = _split_base_and_slice(split)

    qrels_ds = _load_with_revision_fallback(
        path=dataset_name,
        name=qrels_subset,
        split=qrels_base,
        cache_dir=cache_dir,
        revision=revision,
    )

    qrels_qids_all: List[str] = []
    rels: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for row in qrels_ds:
        qid = row.get("query-id") or row.get("query_id") or row.get("_id") or row.get("id")
        did = row.get("corpus-id") or row.get("corpus_id") or row.get("doc_id") or row.get("id")
        score = row.get("score", row.get("relevance", row.get("label", 0)))

        if qid is None or did is None:
            continue

        qid = str(qid)
        if not qrels_qids_all or qrels_qids_all[-1] != qid:
            qrels_qids_all.append(qid)

        try:
            score_f = float(score)
        except Exception:
            continue

        if score_f >= float(min_rel_score):
            rels[qid].append((f"{doc_id_prefix}{did}", score_f))

    seen = set()
    qrels_qids = []
    for qid in qrels_qids_all:
        if qid not in seen:
            seen.add(qid)
            qrels_qids.append(qid)

    if qrels_slice is not None:
        qrels_qids = qrels_qids[qrels_slice]
    if limit is not None:
        qrels_qids = qrels_qids[: min(limit, len(qrels_qids))]

    kept_qid_set = set(qrels_qids)

    queries_ds = _load_with_revision_fallback(
        path=dataset_name,
        name=queries_subset,
        split=split,
        cache_dir=cache_dir,
        revision=revision,
    )

    qid_to_text: Dict[str, str] = {}
    for row in queries_ds:
        qid = row.get("_id") or row.get("query-id") or row.get("query_id") or row.get("id")
        if qid is None:
            continue
        if kept_qid_set and str(qid) not in kept_qid_set:
            continue
        qtext = row.get("text") or row.get("query") or row.get("contents") or ""
        qid_to_text[str(qid)] = str(qtext)

    corpus_ds = _load_with_revision_fallback(
        path=dataset_name,
        name=corpus_subset,
        split="corpus",
        cache_dir=cache_dir,
        revision=revision,
    )

    documents: List[DocumentRecord] = []
    for row in corpus_ds:
        raw_id = row.get("_id") or row.get("id")
        if raw_id is None:
            raise ValueError(f"NFCorpus corpus row missing id field. Keys: {list(row.keys())}")

        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        contents = f"{title}\n\n{text}".strip() if title else text

        documents.append(
            DocumentRecord(
                doc_id=f"{doc_id_prefix}{raw_id}",
                contents=contents,
                metadata={"title": title or None, "raw_id": str(raw_id), "dataset": "NFCorpus"},
            )
        )

    queries: List[QueryRecord] = []
    for qid in qrels_qids:
        pairs = sorted(rels.get(qid, []), key=lambda x: x[1], reverse=True)
        relevant_doc_ids = [doc_id for doc_id, _ in pairs]
        qrels_scores = {doc_id: score for doc_id, score in pairs}

        queries.append(
            QueryRecord(
                query_id=f"q.{qid}",
                contents=qid_to_text.get(qid, ""),
                relevant=relevant_doc_ids,
                metadata={"dataset": "NFCorpus", "qrels_scores": qrels_scores},
            )
        )

    return documents, queries
