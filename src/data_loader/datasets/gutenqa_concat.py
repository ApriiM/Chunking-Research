from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

from src.data_loader.datasets._answer_utils import build_unified_answer_metadata, split_text_answers
from src.data_loader.core.registry import dataset
from src.data_loader.core.schemas import DocumentRecord, QueryRecord


def _split_base_and_slice(split_expr: str) -> Tuple[str, Optional[slice]]:
    """
    Parse expressions like 'train[:200]' into ('train', slice(None, 200)).
    Mirrors the helper used in the base gutenqa loader.
    """
    if not split_expr or "[" not in split_expr or not split_expr.endswith("]"):
        return split_expr, None

    base, bracket = split_expr.split("[", 1)
    base = base.strip()
    inner = bracket[:-1].strip()  # drop trailing ']'

    if ":" not in inner:
        return base, None

    left, right = inner.split(":", 1)
    left = left.strip()
    right = right.strip()

    start = int(left) if left else None
    stop = int(right) if right else None
    return base, slice(start, stop)


@dataset("gutenqa_concat")
def load_gutenqa_concat(
    split: str = None,
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    dataset_name: str = "LumberChunker/GutenQA",
    paragraphs_subset: str = "gutenqa",
    queries_subset: str = "questions",
    revision: Optional[str] = None,
) -> Tuple[List[DocumentRecord], List[QueryRecord]]:
    """
    Variant of the GutenQA loader that:
      1) groups paragraph chunks by their base id (drops the trailing '_<chunk>'),
         concatenating chunk texts in chunk-id order into one long document per group;
      2) rewrites query relevants to the grouped id (e.g., gutenqa_doc_0_12 -> gutenqa_doc_0).
    """

    # ----- Load paragraph chunks -----
    ds_paragraphs = load_dataset(
        path=dataset_name,
        name=paragraphs_subset,
        split="gutenqa_chunks",
        cache_dir=cache_dir,
        revision=revision,
    )

    grouped_chunks: Dict[str, List[Tuple[int, str]]] = {}

    for row in ds_paragraphs:
        book_id = row.get("Book ID")
        chunk_id = row.get("Chunk ID")
        text = row.get("Chunk", "")

        base_id = f"gutenqa_doc_{book_id}"
        try:
            order_idx = int(chunk_id)
        except Exception:
            # Fallback: append in arrival order if chunk id is missing/non-numeric
            order_idx = len(grouped_chunks.get(base_id, []))

        grouped_chunks.setdefault(base_id, []).append((order_idx, text))

    documents: List[DocumentRecord] = []
    for base_id, parts in grouped_chunks.items():
        parts_sorted = sorted(parts, key=lambda x: x[0])
        contents = "\n\n".join(p for _, p in parts_sorted if p)
        documents.append(
            DocumentRecord(
                doc_id=base_id,
                contents=contents,
                metadata={},
            )
        )

    existing_ids = {doc.doc_id for doc in documents}
    doc_contents_by_id = {doc.doc_id: doc.contents for doc in documents}

    # ----- Load questions / queries -----
    base_split, slice_obj = _split_base_and_slice(split)

    slice_str = ""
    if slice_obj is not None:
        left = "" if slice_obj.start is None else slice_obj.start
        right = "" if slice_obj.stop is None else slice_obj.stop
        slice_str = f"[{left}:{right}]"

    ds_questions = load_dataset(
        path=dataset_name,
        name=queries_subset,
        split=f"gutenqa_questions{slice_str}",
        cache_dir=cache_dir,
        revision=revision,
    )

    if limit is not None:
        ds_questions = ds_questions.select(range(min(limit, len(ds_questions))))

    queries: List[QueryRecord] = []
    for idx, row in enumerate(ds_questions):
        orig_relevant = f"gutenqa_doc_{row.get('Book ID')}_{row.get('Chunk ID')}"
        base_relevant = orig_relevant.rsplit("_", 1)[0]

        if base_relevant not in existing_ids:
            continue  # skip queries whose grouped doc is missing

        answer = row.get("Answer")
        extractive_answers, free_text_answers = split_text_answers(
            doc_contents_by_id.get(base_relevant, ""),
            [answer],
        )
        queries.append(
            QueryRecord(
                query_id=f"q.{idx + 1}",
                contents=row.get("Question", "") or "",
                relevant=[base_relevant],
                metadata=build_unified_answer_metadata(
                    base_metadata={"Answer": answer},
                    extractive_answers=extractive_answers,
                    free_text_answers=free_text_answers,
                ),
            )
        )

    return documents, queries
