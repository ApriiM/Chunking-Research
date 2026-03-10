import re
from typing import List, Dict, Any, Optional

from ..core.base import BaseChunker, Chunk
from ..core.progress import coerce_progress_enabled, iter_with_progress
from ..core.registry import chunker


@chunker("passage_regexp")
@chunker("passage")
class RegexpPassageChunker(BaseChunker):
    """
    Groups text into passages of N sentences using a regex-based sentence splitter.

    Config options (merged with configs/chunkers/passage_regexp.yaml defaults):
        passage_length (int): Sentences per chunk; must be > 0. Setting passage_length=1 yields
            one sentence per chunk (i.e., sentence-level segmentation).

    Registered as "passage_regexp"; the "passage" alias is kept for compatibility.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RegexpPassageChunker with passage length from config.

        :param config: Configuration dictionary with 'passage_length' (int, >0)
        :type config: Optional[Dict[str, Any]]
        """

        super().__init__(config)
        self.passage_length: int = int(self.config["passage_length"])
        if self.passage_length <= 0:
            raise ValueError("passage_length must be positive")
        
        # Split sentences on whitespace that follows ., !, or ?.
        self._splitter = re.compile(r"(?<=[.!?])\s+")

    def split_text(
        self,
        documents: List[str],
        documents_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Chunk]:
        # Ensure per-document metadata aligns with the input list.
        if documents_meta is not None and len(documents_meta) != len(documents):
            raise ValueError("documents_meta length must match documents length")

        show_progress = coerce_progress_enabled(self.config.get("show_progress"), default=True)
        all_chunks: List[Chunk] = []
        for idx, text in enumerate(
            iter_with_progress(documents, desc="PassageRegexp Chunking", enabled=show_progress)
        ):
            meta = documents_meta[idx] if documents_meta is not None else None
            all_chunks.extend(self._split_single(text, meta))
        return all_chunks

    def _split_single(
        self, text: str, document_meta: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        document_meta = document_meta or {}
        if not text:
            return []

        # Sentence tokenize first, then group into fixed-size passages.
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        for start_idx in range(0, len(sentences), self.passage_length):
            group = sentences[start_idx : start_idx + self.passage_length]
            if not group:
                continue
            chunk_text = " ".join(group)
            end_idx = start_idx + len(group)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **document_meta,
                        "start_sentence": start_idx,
                        "end_sentence": end_idx,
                    },
                )
            )
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split the input text into sentences using regex.
        """
        # Trim and drop empty segments after splitting.
        parts = self._splitter.split(text.strip()) if text else []
        return [p.strip() for p in parts if p.strip()]
