"""Shared lightweight data schemas for chunking outputs."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from src.chunking.core.base import Chunk


@dataclass
class ChunkRecord:
    """Serializable representation of a `Chunk` plus metadata."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]

    @classmethod
    def from_chunk(cls, chunk: Chunk, sample_id: Optional[str] = None) -> "ChunkRecord":
        """Build a `ChunkRecord` from a `Chunk` while preserving metadata."""

        merged_meta = dict(chunk.metadata or {})
        if sample_id is not None and "sample_id" not in merged_meta:
            merged_meta["sample_id"] = sample_id
        return cls(chunk_id=str(chunk.chunk_id), text=chunk.text, metadata=merged_meta)

    def to_chunk(self) -> Chunk:
        """Rehydrate this record into a `Chunk` instance."""

        return Chunk(text=self.text, chunk_id=self.chunk_id, metadata=self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation."""

        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChunkRecord":
        """Construct a `ChunkRecord` from its JSON form."""

        return cls(
            chunk_id=str(payload.get("chunk_id") or uuid.uuid4()),
            text=payload.get("text", ""),
            metadata=payload.get("metadata", {}) or {},
        )


def save_chunk_records_jsonl(records: Iterable[ChunkRecord], path: str) -> None:
    """Persist chunk records to JSONL."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_chunk_records_jsonl(path: str) -> List[ChunkRecord]:
    """Load chunk records from JSONL into `ChunkRecord` instances."""

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    loaded: List[ChunkRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            loaded.append(ChunkRecord.from_dict(json.loads(line)))
    return loaded
