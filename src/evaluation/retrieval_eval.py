from typing import Dict, List, Optional, Sequence, Tuple

from src.chunking.base import Chunk
from src.data_loader.types import QASample
from src.evaluation.retrieval import RetrievalResult, TfidfRetriever, evaluate_retrieval


def retrieval_at_k(
    samples: Sequence[QASample],
    chunks: Sequence[Chunk],
    retriever: Optional[TfidfRetriever] = None,
    top_k: int = 5,
    relevance: str = "substring",
) -> Tuple[Dict[str, float], List[RetrievalResult]]:
    """Evaluate retrieval given samples and precomputed chunks.

    If chunks are precomputed, we reuse them; otherwise, caller should run chunker beforehand.
    """
    retriever = retriever or TfidfRetriever()
    # We expect chunk metadata to already contain sample_id; group by sample_id
    chunks_by_sample: Dict[str, List[Chunk]] = {}
    for c in chunks:
        sid = c.metadata.get("sample_id") if c.metadata else None
        if sid is None:
            continue
        chunks_by_sample.setdefault(sid, []).append(c)

    subset_samples: List[QASample] = []
    subset_chunks: List[List[Chunk]] = []
    for s in samples:
        subset_samples.append(s)
        subset_chunks.append(chunks_by_sample.get(s.sample_id, []))

    # Reuse the evaluate_retrieval core by temporarily swapping split_text behavior
    class _Prechunked:
        def __init__(self, per_sample: List[List[Chunk]]):
            self.per_sample = per_sample
            self._i = 0

        def split_text(self, _text: str, document_meta=None):
            out = self.per_sample[self._i]
            self._i += 1
            return out

    dummy_chunker = _Prechunked(subset_chunks)
    metrics, results, _ = evaluate_retrieval(
        subset_samples, dummy_chunker, retriever=retriever, top_k=top_k, relevance=relevance, return_chunks=False
    )
    return metrics, results
