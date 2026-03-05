from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from src.data_loader.datasets._answer_utils import build_unified_answer_metadata, split_text_answers
from src.data_loader.core.registry import dataset
from src.data_loader.core.schemas import DocumentRecord, QueryRecord


# This should be moved to a common utils file
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


@dataset("gutenqa")
def load_gutenqa(
    split: str = None,
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    dataset_name: str = "LumberChunker/GutenQA",
    paragraphs_subset: str = "gutenqa",
    queries_subset : str = "questions",
    revision: Optional[str] = None,
) -> Tuple[List[DocumentRecord], List[QueryRecord]]:
    
    documents: List[DocumentRecord] = []
    queries: List[QueryRecord] = []
    doc_contents_by_id: Dict[str, str] = {}

    ds_paragraphs = load_dataset(
        path=dataset_name,
        name=paragraphs_subset,
        split="gutenqa_chunks",
        cache_dir=cache_dir,
        revision=revision
    )
    
    for row in ds_paragraphs:
        doc_id = f"gutenqa_doc_{row.get('Book ID')}_{row.get('Chunk ID')}"
        doc_meta: Dict[str, object] = {}
        doc_meta["Book Name"] = row.get("Book Name")
        doc_meta["Book Chapter"] = row.get("Chapter")

        documents.append(
            DocumentRecord(
                doc_id=doc_id,
                contents=row.get("Chunk"),
                metadata=doc_meta
            )
        )
        doc_contents_by_id[doc_id] = row.get("Chunk") or ""
    
    _, slice_obj = _split_base_and_slice(split)

    slice_str = ""
    if slice_obj is not None:
        slice_str = f'[{slice_obj.start if slice_obj.start is not None else ""}:{slice_obj.stop if slice_obj.stop is not None else ""}]'

    ds_questions = load_dataset(
        path=dataset_name,
        name=queries_subset,
        split=f"gutenqa_questions{slice_str}",
        cache_dir=cache_dir,
        revision=revision
    )

    ds_questions = ds_questions.select(range(min(limit, len(ds_questions)))) if limit is not None else ds_questions

    for idx, row in enumerate(ds_questions):
        relevant_doc_id = f"gutenqa_doc_{row.get('Book ID')}_{row.get('Chunk ID')}"

        if relevant_doc_id in doc_contents_by_id:
            answer = row.get("Answer")
            extractive_answers, free_text_answers = split_text_answers(
                doc_contents_by_id[relevant_doc_id],
                [answer],
            )
            query_meta = build_unified_answer_metadata(
                base_metadata={"Answer": answer},
                extractive_answers=extractive_answers,
                free_text_answers=free_text_answers,
            )
            queries.append(
                QueryRecord(
                    query_id=f"q.{idx + 1}",
                    contents=row.get("Question", "") or "",
                    relevant=[relevant_doc_id],
                    metadata=query_meta
                )
            )
        else:
            print(f"Warning: Relevant document {relevant_doc_id} for question index {idx} not found in documents.")
    
    return documents, queries
    
