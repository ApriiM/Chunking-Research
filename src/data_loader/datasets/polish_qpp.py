from typing import List, Optional

from datasets import load_dataset

from src.data_loader.core.types import QASample
from src.data_loader.core.registry import dataset


@dataset("polish_qpp")
def load_polish_qpp(
    split: str = "train",
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    revision: Optional[str] = None,
    positives_only: bool = True,
) -> List[QASample]:
    """Load Allegro Polish Question-Passage Pairs dataset.

    Each row contains a question, a Wikipedia passage, and a label (`correct`) indicating
    if the passage answers the question. Answers are not provided; we surface the passage
    as context and optionally drop negative pairs.

    :param split: Dataset split or slice (e.g., "train", "train[:200]").
    :param cache_dir: Optional HF cache directory.
    :param limit: Optional cap on returned rows after filtering.
    :param revision: Optional HF revision/tag/commit.
    :param positives_only: If True, keep only rows with correct==1.
    """
    dataset = load_dataset(
        "allegro/polish-question-passage-pairs",
        split=split,
        cache_dir=cache_dir,
        revision=revision,
    )

    if positives_only:
        dataset = dataset.filter(lambda row: int(row.get("correct", 0)) == 1)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    samples: List[QASample] = []
    for idx, row in enumerate(dataset):
        sample_id = f"{split}-{idx}"
        samples.append(
            QASample(
                sample_id=sample_id,
                context=row.get("passage", ""),
                question=row.get("question", ""),
                answers=[],
                answer_starts=[],
                title=row.get("article_title"),
            )
        )
    return samples
