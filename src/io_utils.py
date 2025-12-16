import json
import os
from datetime import datetime
from typing import Iterable, Optional


def save_jsonl(records: Iterable[dict], path: str, overwrite: bool = False) -> str:
    if not path:
        return ""
    if os.path.exists(path) and not overwrite:
        base, ext = os.path.splitext(path)
        path = f"{base}_{datetime.now().strftime('%Y%m%dT%H%M%S')}{ext or '.jsonl'}"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_manifest(path: str, payload: dict, overwrite: bool = True) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path) and not overwrite:
        base, ext = os.path.splitext(path)
        path = f"{base}_{datetime.now().strftime('%Y%m%dT%H%M%S')}{ext or '.json'}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
