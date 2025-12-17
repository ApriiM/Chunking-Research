from typing import Dict, Any, List, Optional

from nltk.tokenize import TextTilingTokenizer
from nltk.tokenize.texttiling import (
    BLOCK_COMPARISON,
    VOCABULARY_INTRODUCTION,
    DEFAULT_SMOOTHING,
    HC,
    LC,
)

from ..core.base import BaseChunker, Chunk
from ..core.registry import chunker


@chunker("text_tiling")
class TextTilingChunker(BaseChunker):
    """Wraps NLTK's TextTilingTokenizer for text segmentation.

    This follows the implementation shipped in NLTK (originating from Hearst, 1997),
    not a re-implementation tailored to newer variants. Parameters are passed
    through from config.

    Original paper: https://aclanthology.org/J97-1003.pdf

    Original code: https://www.nltk.org/_modules/nltk/tokenize/texttiling.html
    Documentation: https://www.nltk.org/api/nltk.tokenize.texttiling.html
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.tokenizer = TextTilingTokenizer(
            w=int(self.config["w"]),
            k=int(self.config["k"]),
            similarity_method=self._resolve_similarity(self.config.get("similarity_method")),
            stopwords=self._resolve_stopwords(self.config.get("stopwords")),
            smoothing_method=self._resolve_smoothing(self.config.get("smoothing_method")),
            smoothing_width=int(self.config["smoothing_width"]),
            smoothing_rounds=int(self.config["smoothing_rounds"]),
            cutoff_policy=self._resolve_cutoff(self.config.get("cutoff_policy")),
            demo_mode=bool(self.config.get("demo_mode", False)),
        )

    def split_text(
        self, text: str, document_meta: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        document_meta = document_meta or {}
        if not text:
            return []

        segments = self.tokenizer.tokenize(text)
        chunks: List[Chunk] = []
        for idx, seg in enumerate(segments):
            seg_text = seg.strip()
            if not seg_text:
                continue
            chunks.append(
                Chunk(
                    text=seg_text,
                    metadata={
                        **document_meta,
                        "segment_index": idx,
                    },
                )
            )
        return chunks

    def _resolve_similarity(self, value: Any):
        mapping = {
            "block_comparison": BLOCK_COMPARISON,
            "vocabulary_introduction": VOCABULARY_INTRODUCTION,
            BLOCK_COMPARISON: BLOCK_COMPARISON,
            VOCABULARY_INTRODUCTION: VOCABULARY_INTRODUCTION,
        }
        return mapping.get(value, BLOCK_COMPARISON)

    def _resolve_smoothing(self, value: Any):
        if value is None:
            return DEFAULT_SMOOTHING
        if isinstance(value, str):
            if value == "default":
                return DEFAULT_SMOOTHING
            # Unknown string falls back to default
            return DEFAULT_SMOOTHING
        # Allow callers to pass explicit sequences (e.g., list/tuple weights)
        return value

    def _resolve_cutoff(self, value: Any):
        mapping = {
            "hc": HC,
            "lc": LC,
            HC: HC,
            LC: LC,
        }
        return mapping.get(value, HC)

    def _resolve_stopwords(self, value: Any):
        # If None, avoid NLTK download requirement by using empty list
        if value is None or value == "None":
            return None
        return value
