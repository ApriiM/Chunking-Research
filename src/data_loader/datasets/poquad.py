from typing import List, Optional

from datasets import load_dataset

from src.data_loader.core.types import QASample
from src.data_loader.core.registry import dataset


@dataset("poquad")
def load_poquad(
    split: str = "train",
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    revision: str = "refs/convert/parquet",
) -> List[QASample]:
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

    samples: List[QASample] = []
    for row in dataset:
        answers = row.get("answers") or {}
        answer_texts = list(answers.get("text", []) or [])
        answer_starts = list(answers.get("answer_start", []) or [])
        samples.append(
            QASample(
                sample_id=str(row.get("id")),
                context=row.get("context", ""),
                question=row.get("question", ""),
                answers=answer_texts,
                answer_starts=answer_starts,
                title=row.get("title"),
            )
        )
    return samples
