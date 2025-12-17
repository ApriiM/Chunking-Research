from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QASample:
    """Minimal QA example shared across loaders and experiments."""

    sample_id: str
    context: str
    question: str
    answers: List[str]
    answer_starts: Optional[List[int]] = None
    title: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "context": self.context,
            "question": self.question,
            "answers": self.answers,
            "answer_starts": self.answer_starts,
            "title": self.title,
        }

    @staticmethod
    def from_dict(payload: dict) -> "QASample":
        return QASample(
            sample_id=str(payload.get("sample_id")),
            context=payload.get("context", ""),
            question=payload.get("question", ""),
            answers=list(payload.get("answers", []) or []),
            answer_starts=payload.get("answer_starts"),
            title=payload.get("title"),
        )
