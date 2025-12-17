import json
import os
from typing import Iterable, List

from src.data_loader.core.types import QASample


def save_samples_jsonl(samples: Iterable[QASample], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


def load_samples_jsonl(path: str) -> List[QASample]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    records: List[QASample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            records.append(QASample.from_dict(payload))
    return records
