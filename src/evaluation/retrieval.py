from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking.base import BaseChunker, Chunk
from src.data_loader.types import QASample


@dataclass
class RetrievalResult:
    sample_id: str
    question: str
    answers: List[str]
    relevant_chunk_ids: List[str]
    retrieved_chunk_ids: List[str]
    scores: List[float]


class TfidfRetriever:
    """Simple TF-IDF + cosine retriever used as a baseline."""

    def rank(self, query: str, chunk_texts: Sequence[str]) -> List[float]:
        if not chunk_texts:
            return []
        # Fit per-query for clarity; swap in a shared vectorizer for speed on larger corpora.
        vectorizer = TfidfVectorizer()
        corpus = [query] + list(chunk_texts)
        matrix = vectorizer.fit_transform(corpus)
        query_vec = matrix[0:1]
        chunk_vecs = matrix[1:]
        scores = cosine_similarity(query_vec, chunk_vecs).flatten()
        return scores.tolist()


def _relevant_chunk_ids(answers: Sequence[str], chunks: Sequence[Chunk]) -> List[str]:
    """Identify chunks that contain any reference answer (substring match)."""
    relevant = []
    normalized_answers = [a.lower() for a in answers if a]
    for chunk in chunks:
        chunk_text = chunk.text.lower()
        if any(ans in chunk_text for ans in normalized_answers):
            relevant.append(chunk.chunk_id)
    return relevant


def evaluate_retrieval(
    samples: Sequence[QASample],
    chunker: BaseChunker,
    retriever: Optional[TfidfRetriever] = None,
    top_k: int = 5,
    relevance: str = "substring",
    return_chunks: bool = False,
) -> Tuple[Dict[str, float], List[RetrievalResult], Optional[List[dict]]]:
    """Run chunking + retrieval evaluation over QA samples.

    :param samples: QA examples (context, question, answers)
    :param chunker: Chunker instance implementing split_text
    :param retriever: Retriever used to rank chunks for each question
    :param top_k: Number of chunks to retrieve per question
    :param relevance: Policy for marking relevant chunks (currently substring only)
    :param return_chunks: Whether to also return serialized chunk records
    :return: (aggregate metrics, per-sample results, optional chunk records)
    """
    retriever = retriever or TfidfRetriever()

    results: List[RetrievalResult] = []
    chunk_records: List[dict] = []
    f1_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    hits: List[int] = []

    for sample in samples:
        # Chunk each sample context once; chunk metadata carries the originating sample id.
        chunks = chunker.split_text(sample.context, document_meta={"sample_id": sample.sample_id})
        if return_chunks:
            for chunk in chunks:
                chunk_records.append(
                    {
                        "sample_id": sample.sample_id,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    }
                )
        chunk_texts = [c.text for c in chunks]
        scores = retriever.rank(sample.question, chunk_texts)

        sorted_indices = np.argsort(scores)[::-1]
        top_indices = sorted_indices[:top_k]
        retrieved_ids = [chunks[i].chunk_id for i in top_indices] if len(chunks) else []

        if relevance == "substring":
            relevant_ids = _relevant_chunk_ids(sample.answers, chunks)
        else:
            # Default fallback ensures metrics remain defined until new policies are added.
            relevant_ids = _relevant_chunk_ids(sample.answers, chunks)

        tp = len(set(retrieved_ids) & set(relevant_ids))
        denom = len(retrieved_ids) if retrieved_ids else top_k
        precision = tp / denom if denom else 0.0
        recall = tp / len(relevant_ids) if relevant_ids else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        hit = 1 if tp > 0 else 0

        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        hits.append(hit)

        result = RetrievalResult(
            sample_id=sample.sample_id,
            question=sample.question,
            answers=sample.answers,
            relevant_chunk_ids=relevant_ids,
            retrieved_chunk_ids=retrieved_ids,
            scores=[scores[i] for i in top_indices] if len(scores) else [],
        )
        results.append(result)

    metrics = {
        "f1_at_k": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "hit_rate_at_k": float(np.mean(hits)) if hits else 0.0,
    }
    return metrics, results, chunk_records if return_chunks else None
