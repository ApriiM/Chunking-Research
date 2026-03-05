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
from ..core.progress import coerce_progress_enabled, iter_with_progress
from ..core.registry import chunker


@chunker("text_tiling")
class TextTilingChunker(BaseChunker):
    """Wraps NLTK's TextTilingTokenizer for text segmentation.

    This follows the implementation shipped in NLTK (originating from Hearst, 1997),
    not a re-implementation tailored to newer variants. Parameters are passed
    through from config.

    Config options (merged with configs/chunkers/text_tiling.yaml defaults):
        w (int): Token-sequence size (pseudosentence length); must be > 0.
        k (int): Number of token-sequences per block; must be > 0.
        similarity_method (str or constant): "block_comparison" or "vocabulary_introduction".
        stopwords (List[str] or None): Stopword list passed to NLTK.
        smoothing_method (str or tuple): "default" or an NLTK smoothing method.
        smoothing_width (int): Smoothing window width; must be > 0.
        smoothing_rounds (int): Number of smoothing passes; must be > 0.
        cutoff_policy (str or constant): "hc" or "lc".
        demo_mode (bool): Enable NLTK demo/debug mode.

    Original paper: https://aclanthology.org/J97-1003.pdf
    Original code: https://www.nltk.org/_modules/nltk/tokenize/texttiling.html
    Documentation: https://www.nltk.org/api/nltk.tokenize.texttiling.html
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Map config values to NLTK's TextTilingTokenizer arguments.
        self.tokenizer = TextTilingTokenizer(
            w=self._as_int("w"),
            k=self._as_int("k"),
            similarity_method=self._resolve_similarity(self.config.get("similarity_method")),
            stopwords=self._resolve_stopwords(self.config.get("stopwords")),
            smoothing_method=self._resolve_smoothing(self.config.get("smoothing_method")),
            smoothing_width=self._as_int("smoothing_width"),
            smoothing_rounds=self._as_int("smoothing_rounds"),
            cutoff_policy=self._resolve_cutoff(self.config.get("cutoff_policy")),
            demo_mode=bool(self.config.get("demo_mode", False)),
        )

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
            iter_with_progress(documents, desc="TextTiling Chunking", enabled=show_progress)
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

        # TextTiling can raise on short/degenerate inputs; fall back to one segment.
        try:
            segments = self.tokenizer.tokenize(text)
        except ValueError as exc:
            print(f"TextTiling failed; returning full text as a single segment. error={exc}")
            segments = [text]

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
        # Accept either string labels or NLTK constants.
        mapping = {
            "block_comparison": BLOCK_COMPARISON,
            "vocabulary_introduction": VOCABULARY_INTRODUCTION,
            BLOCK_COMPARISON: BLOCK_COMPARISON,
            VOCABULARY_INTRODUCTION: VOCABULARY_INTRODUCTION,
        }
        return mapping.get(value, BLOCK_COMPARISON)

    def _resolve_smoothing(self, value: Any):
        # Use the NLTK default unless a custom smoothing method is provided.
        if value is None:
            return DEFAULT_SMOOTHING
        if isinstance(value, str):
            if value == "default":
                return DEFAULT_SMOOTHING
            return DEFAULT_SMOOTHING
        return value

    def _resolve_cutoff(self, value: Any):
        # Accept either string labels or NLTK constants.
        mapping = {
            "hc": HC,
            "lc": LC,
            HC: HC,
            LC: LC,
        }
        return mapping.get(value, HC)

    def _resolve_stopwords(self, value: Any):
        # Treat None/"None" as no stopword filtering.
        if value is None or value == "None":
            return None
        return value

    def _as_int(self, key: str) -> int:
        # Validate required integer parameters from config.
        val = int(self.config[key])
        if val <= 0:
            raise ValueError(f"{key} must be positive")
        return val
