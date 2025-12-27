from typing import List, Dict, Any, Optional

from ..core.base import BaseChunker, Chunk
from ..core.registry import chunker


@chunker("fixed_size")
class FixedSizeChunker(BaseChunker):
    '''
    Splits text into fixed-size chunks with optional overlap.
    '''
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        '''
        Initialize FixedSizeChunker with chunk size and overlap from config.

        :param config: Configuration dictionary with 'chunk_size' and 'overlap' keys
        :type config: Optional[Dict[str, Any]]
        '''
        super().__init__(config)
        self.chunk_size = int(self.config["chunk_size"])
        self.overlap = int(self.config["overlap"])

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size to avoid infinite loops")

    def split_text(
        self,
        documents: List[str],
        documents_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Chunk]:
        if documents_meta is not None and len(documents_meta) != len(documents):
            raise ValueError("documents_meta length must match documents length")

        all_chunks: List[Chunk] = []
        for idx, text in enumerate(documents):
            meta = documents_meta[idx] if documents_meta is not None else None
            all_chunks.extend(self._split_single(text, meta))
        return all_chunks

    def _split_single(
        self, text: str, document_meta: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        document_meta = document_meta or {}
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            new_chunk = Chunk(
                text=chunk_text,
                metadata={
                    **document_meta,
                    "start_char": start,
                    "end_char": end,
                },
            )
            chunks.append(new_chunk)

            if end == len(text):
                break

            start += self.chunk_size - self.overlap

        return chunks
