from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.chunking.base import Chunk
from src.data_loader.types import QASample
from src.evaluation.retrieval import RetrievalResult

SCHEMA_VERSION = "1.0"


def _require(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


@dataclass
class SampleRecord:
    sample_id: str
    context: str
    question: str
    answers: List[str]
    answer_starts: Optional[List[int]] = None
    title: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    @staticmethod
    def from_sample(sample: QASample) -> "SampleRecord":
        return SampleRecord(
            sample_id=sample.sample_id,
            context=sample.context,
            question=sample.question,
            answers=list(sample.answers),
            answer_starts=sample.answer_starts,
            title=sample.title,
        )


@dataclass
class ChunkRecord:
    sample_id: str
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    @staticmethod
    def from_chunk(chunk: Chunk, sample_id: Optional[str] = None) -> "ChunkRecord":
        return ChunkRecord(
            sample_id=sample_id or chunk.metadata.get("sample_id", ""),
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata=chunk.metadata,
        )


@dataclass
class RetrievalResultRecord:
    sample_id: str
    question: str
    answers: List[str]
    relevant_chunk_ids: List[str]
    retrieved_chunk_ids: List[str]
    scores: List[float]
    schema_version: str = SCHEMA_VERSION

    @staticmethod
    def from_result(result: RetrievalResult) -> "RetrievalResultRecord":
        return RetrievalResultRecord(
            sample_id=result.sample_id,
            question=result.question,
            answers=list(result.answers),
            relevant_chunk_ids=list(result.relevant_chunk_ids),
            retrieved_chunk_ids=list(result.retrieved_chunk_ids),
            scores=list(result.scores),
        )


def validate_samples(samples: List[QASample]) -> None:
    for s in samples:
        _require(s.sample_id is not None, "sample_id missing")
        _require(isinstance(s.answers, list), "answers must be list")
        _require(s.context is not None, "context missing")
        _require(s.question is not None, "question missing")


def validate_chunks(chunks: List[Any]) -> None:
    for c in chunks:
        chunk_id = getattr(c, "chunk_id", None) if not isinstance(c, dict) else c.get("chunk_id")
        text = getattr(c, "text", None) if not isinstance(c, dict) else c.get("text")
        _require(chunk_id is not None, "chunk_id missing")
        _require(text is not None, "chunk text missing")


def validate_retrieval_results(results: List[RetrievalResult]) -> None:
    for r in results:
        _require(r.sample_id is not None, "retrieval result sample_id missing")
        _require(isinstance(r.retrieved_chunk_ids, list), "retrieved_chunk_ids must be list")
