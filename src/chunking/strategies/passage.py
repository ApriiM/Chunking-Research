import re
from typing import List, Dict, Any, Optional

from ..base import BaseChunker, Chunk


class SentencePassageChunker(BaseChunker):
    """
    Groups text into passages of N sentences, similar to the passage_length
    preprocessing used in HeterGraphLongSum.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        '''
        Initialize SentencePassageChunker with passage length from config.
        
        :param config: Configuration dictionary with 'passage_length' key
        :type config: Optional[Dict[str, Any]]
        ''' 
        super().__init__(config)
        self.passage_length = int(self.config["passage_length"])
        if self.passage_length <= 0:
            raise ValueError("passage_length must be positive")
        self._splitter = re.compile(r"(?<=[.!?])\s+")

    def split_text(
        self, text: str, document_meta: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        document_meta = document_meta or {}
        sentences = self._split_sentences(text)
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
        '''
        Split the input text into sentences using regex.
        
        :param text: Input text to split into sentences
        :type text: str
        :return: List of sentences
        :rtype: List[str]
        ''' 
        if not text:
            return []
        parts = self._splitter.split(text.strip())
        return [p.strip() for p in parts if p.strip()]
