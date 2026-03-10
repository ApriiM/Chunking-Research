from typing import List, Dict, Any, Optional

from ..core.base import BaseChunker, Chunk
from ..core.progress import coerce_progress_enabled, iter_with_progress
from ..core.registry import chunker


@chunker("fixed_size")
class FixedSizeChunker(BaseChunker):
    '''
    Splits text into fixed-size character chunks with optional overlap.

    Config options (merged with configs/chunkers/fixed_size.yaml defaults when using registry):
        chunk_size (int): Number of characters per chunk; must be > 0.
        overlap (int): Number of characters to overlap between chunks; 0 <= overlap < chunk_size.
    '''

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        '''
        Initialize FixedSizeChunker with chunk size and overlap from config.

        :param config: Configuration dictionary with:
            - chunk_size: int, number of characters per chunk
            - overlap: int, number of characters shared between adjacent chunks
        :type config: Optional[Dict[str, Any]]
        '''
        super().__init__(config)
        self.chunk_size: int = int(self.config["chunk_size"])
        self.overlap: int = int(self.config["overlap"])

        # Validate to avoid degenerate or infinite windowing.
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size to avoid infinite loops")
        
        # Step size for the sliding window; overlap reduces the step.
        self._step: int = self.chunk_size - self.overlap

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
            iter_with_progress(documents, desc="FixedSize Chunking", enabled=show_progress)
        ):
            meta = documents_meta[idx] if documents_meta is not None else None
            all_chunks.extend(self._split_single(text, meta))
        return all_chunks

    def _split_single(
            self, text: str, document_meta: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        if not text:
            return []

        document_meta = document_meta or {}
        text_len = len(text)

        step = self._step
        chunk_size = self.chunk_size
        chunks: List[Chunk] = []
        chunks_append = chunks.append

        # Slide a fixed-size window over the raw character string.
        start = 0
        while start < text_len:
            end = start + chunk_size
            if end > text_len:
                end = text_len

            meta = document_meta.copy()
            meta["start_char"] = start
            meta["end_char"] = end

            # Store character offsets so downstream tools can re-map spans.
            chunks_append(
                Chunk(
                    text=text[start:end],
                    metadata=meta,
                )
            )
            # Stop once we've emitted the tail chunk.
            if end == text_len:
                break

            start += step

        return chunks