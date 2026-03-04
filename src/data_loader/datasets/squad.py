import hashlib
import random
from typing import Dict, List, Optional, Tuple, Set

from datasets import load_dataset

from src.data_loader.core.registry import dataset
from src.data_loader.core.schemas import DocumentRecord, QueryRecord


_TRAIN_QUERY_SAMPLE_SIZE = 10_000
_TRAIN_QUERY_SAMPLE_SEED = 42


@dataset("squad")
def load_squad(
    split: str = "train",
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    # Prefer the dataset *data repo* (stable, works well with parquet conversion)
    dataset_name: str = "rajpurkar/squad",
    # Match your PoQuAD pattern: avoid dataset scripts when possible
    revision: Optional[str] = "refs/convert/parquet",
    # Some HF dataset repos have configs; SQuAD’s is typically "plain_text"
    config_name: Optional[str] = "default",
    # Useful if you later point dataset_name="squad_v2"
    skip_impossible: bool = False,
) -> Tuple[List[DocumentRecord], List[QueryRecord]]:
    """Load SQuAD-style data from Hugging Face Hub into (documents, queries).

    Notes:
    - HF SQuAD has splits: train, validation (no public test split).
    - Uses parquet conversion ref by default (revision="refs/convert/parquet").
    """

    load_kwargs = dict(
        path=dataset_name,
        split=split,
        cache_dir=cache_dir,
        revision=revision,
    )
    # Only pass config if explicitly set (safe for repo datasets like rajpurkar/squad)
    if config_name is not None:
        load_kwargs["name"] = config_name

    ds = load_dataset(**load_kwargs)

    documents: List[DocumentRecord] = []
    queries: List[QueryRecord] = []

    # Deduplicate documents by identical context to avoid many copies per question.
    context_to_doc: Dict[str, str] = {}

    for row in ds:
        if skip_impossible and row.get("is_impossible") is True:
            continue
        
        title = row.get("title")
        if not title:
            continue

        context = row.get("context") or ""

        answers = row.get("answers") or {}
        answer_texts = list(answers.get("text", []) or [])
        answer_starts = list(answers.get("answer_start", []) or [])

        if title not in context_to_doc.keys():
            context_to_doc[title] = ""

        context_start_id = context_to_doc[title].find(context)
        if context_start_id == -1:
            context_start_id = len(context_to_doc[title])
            context_to_doc[title] += context

        answer_starts = [start + context_start_id for start in answer_starts]

        query_meta: Dict[str, object] = {}
        if answer_texts:
            query_meta["answers"] = answer_texts
        if answer_starts:
            query_meta["answer_starts"] = answer_starts
        if "is_impossible" in row:
            query_meta["is_impossible"] = row.get("is_impossible")

        queries.append(
            QueryRecord(
                query_id=f"q.{row.get('id')}",
                contents=row.get("question", "") or "",
                relevant=[title],
                metadata=query_meta,
            )
        )

    for title, context in context_to_doc.items():
        documents.append(
            DocumentRecord(
                doc_id=title,
                contents=context,
                metadata={},
            )
        )

    if split == "train" and len(queries) > _TRAIN_QUERY_SAMPLE_SIZE:
        rng = random.Random(_TRAIN_QUERY_SAMPLE_SEED)
        sampled_indices = sorted(rng.sample(range(len(queries)), _TRAIN_QUERY_SAMPLE_SIZE))
        queries = [queries[i] for i in sampled_indices]

    if limit is not None:
        queries = queries[: min(limit, len(queries))]

    return documents, queries
