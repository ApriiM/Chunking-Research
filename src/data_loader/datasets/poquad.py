import hashlib
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

from src.data_loader.datasets._answer_utils import build_unified_answer_metadata, dedupe_preserve_order
from src.data_loader.core.registry import dataset
from src.data_loader.core.schemas import DocumentRecord, QueryRecord


@dataset("poquad")
def load_poquad(
    split: str = "train",
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    revision: str = "refs/convert/parquet",
) -> Tuple[List[DocumentRecord], List[QueryRecord]]:
    """Load PoQuAD (SQuAD-style) from Hugging Face Hub.

    :param split: Dataset split or slicing expression (e.g., "train[:500]")
    :param cache_dir: Optional HF cache location
    :param limit: Optional hard cap on number of rows after loading
    :param revision: HF dataset revision (default uses parquet conversion to avoid scripts)
    """
    dataset = load_dataset(
        "clarin-pl/poquad",
        split=split,
        cache_dir=cache_dir,
        revision=revision,
    )

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    documents: List[DocumentRecord] = []
    queries: List[QueryRecord] = []

    # Deduplicate documents by identical context to avoid many copies per question.
    context_to_doc: Dict[str, str] = {}
    doc_meta_store: Dict[str, Dict[str, object]] = {}

    for row in dataset:
        context = row.get("context", "")
        title = row.get("title")
        answers = row.get("answers") or {}
        answer_texts = list(answers.get("text", []) or [])
        answer_starts = list(answers.get("answer_start", []) or [])

        # Stable doc_id derived from context hash for deduplication
        if context not in context_to_doc:
            digest = hashlib.md5(context.encode("utf-8")).hexdigest()[:12]
            doc_id = f"poquad-{digest}"
            context_to_doc[context] = doc_id
            doc_meta: Dict[str, object] = {}
            if title:
                doc_meta["title"] = title
            doc_meta_store[doc_id] = doc_meta
            documents.append(
                DocumentRecord(
                    doc_id=doc_id,
                    contents=context,
                    metadata=doc_meta,
                )
            )
        else:
            doc_id = context_to_doc[context]

        query_meta_base: Dict[str, object] = {}
        if title:
            query_meta_base["title"] = title
        if answer_texts:
            query_meta_base["answers"] = answer_texts
        if answer_starts:
            query_meta_base["answer_starts"] = answer_starts

        query_meta = build_unified_answer_metadata(
            base_metadata=query_meta_base,
            extractive_answers=dedupe_preserve_order(answer_texts),
        )

        queries.append(
            QueryRecord(
                query_id=f"q.{row.get('id')}",
                contents=row.get("question", ""),
                relevant=[doc_id],
                metadata=query_meta,
            )
        )

    return documents, queries
