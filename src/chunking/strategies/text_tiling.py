import re
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


def _load_nltk_stopwords(language: str) -> List[str]:
    from nltk.corpus import stopwords

    return stopwords.words(language)


def _load_polish_stopwords() -> List[str]:
    from spacy.lang.pl.stop_words import STOP_WORDS

    return sorted(STOP_WORDS)


class PseudosentenceBoundaryTextTilingTokenizer(TextTilingTokenizer):
    """NLTK TextTiling with boundaries snapped to pseudosentence ends.

    NLTK's original implementation scores lexical cohesion at pseudosentence
    gaps, then snaps selected gaps to paragraph breaks. This variant keeps the
    scoring pipeline unchanged and replaces only the legal boundary positions:
    selected gaps are normalized to the nearest end of a pseudosentence.
    """

    _NOPUNCT_ALLOWED = re.compile(r"[a-z\-' \n\t]")
    _TRAILING_NONSPACE = re.compile(r"\S")

    def tokenize(self, text):
        """Return TextTiling segments using pseudosentence-end candidates."""

        # Original NLTK step: lowercase, remove punctuation, then divide into
        # fixed-size token sequences ("pseudosentences").
        # Our change: keep an offset map while removing punctuation, because
        # final chunk boundaries must slice the original unmodified text.
        lowercase_text = text.lower()
        text_length = len(lowercase_text)
        nopunct_text, nopunct_to_original = self._remove_punctuation_with_map(
            lowercase_text
        )

        tokseqs = self._divide_to_tokensequences(nopunct_text)
        if len(tokseqs) < 2:
            return [text]

        nopunct_breaks = self._mark_pseudosentence_breaks(tokseqs)
        if len(nopunct_breaks) < 2:
            raise ValueError("No pseudosentence breaks were found")

        # Original NLTK step: find paragraph breaks in original and
        # punctuation-stripped text.
        # Our change: legal boundary candidates are pseudosentence ends. The
        # token table still receives no-punctuation offsets, while final
        # normalization receives equivalent offsets in the original text.
        original_breaks = self._translate_breaks_to_original_text(
            nopunct_breaks,
            nopunct_to_original,
            text,
        )
        if len(original_breaks) < 2:
            raise ValueError("No original-text pseudosentence breaks were found")

        # Original NLTK step: remove configured stopwords from each token
        # sequence before lexical cohesion scoring.
        for ts in tokseqs:
            ts.wrdindex_list = [
                wi for wi in ts.wrdindex_list if wi[0] not in self.stopwords
            ]

        # Original NLTK step: build token table and compute lexical similarity
        # scores between adjacent pseudosentence gaps.
        token_table = self._create_token_table(tokseqs, nopunct_breaks)

        if self.similarity_method == BLOCK_COMPARISON:
            gap_scores = self._block_comparison(tokseqs, token_table)
        elif self.similarity_method == VOCABULARY_INTRODUCTION:
            raise NotImplementedError("Vocabulary introduction not implemented")
        else:
            raise ValueError(
                f"Similarity method {self.similarity_method} not recognized"
            )

        # Our change: short documents that cannot produce enough gap scores for
        # NLTK's smoothing window are returned as one segment without logging a
        # TextTiling failure.
        if len(gap_scores) < self.smoothing_width + 1:
            if self.demo_mode:
                empty_scores = [0 for _ in gap_scores]
                return gap_scores, gap_scores, empty_scores, empty_scores
            return [text]

        # Original NLTK step: smooth gap scores, compute depth scores, and pick
        # boundary indices from strong depth-score valleys.
        if self.smoothing_method == DEFAULT_SMOOTHING:
            smooth_scores = self._smooth_scores(gap_scores)
        else:
            raise ValueError(f"Smoothing method {self.smoothing_method} not recognized")

        depth_scores = self._depth_scores(smooth_scores)
        segment_boundaries = self._identify_boundaries(depth_scores)
        # Original NLTK step: normalize selected boundary indices to nearest
        # legal boundary offset.
        # Our change: the legal offsets are pseudosentence ends, not paragraph
        # breaks, so paragraph formatting no longer controls chunk boundaries.
        normalized_boundaries = self._normalize_boundaries(
            text,
            segment_boundaries,
            original_breaks,
        )

        # Original NLTK step: slice the original text at normalized boundaries.
        segmented_text = []
        prevb = 0
        for b in normalized_boundaries:
            if b == 0:
                continue
            segmented_text.append(text[prevb:b])
            prevb = b

        if prevb < text_length:
            segmented_text.append(text[prevb:])

        if not segmented_text:
            segmented_text = [text]

        if self.demo_mode:
            return gap_scores, smooth_scores, depth_scores, segment_boundaries
        return segmented_text

    def _remove_punctuation_with_map(self, text: str):
        """Our change: NLTK punctuation removal plus offset translation map."""
        chars = []
        offset_map = []
        for idx, char in enumerate(text):
            if self._NOPUNCT_ALLOWED.match(char):
                chars.append(char)
                offset_map.append(idx)
        return "".join(chars), offset_map

    def _mark_pseudosentence_breaks(self, tokseqs):
        """Our change: use ends of scored pseudosentences as legal breaks."""
        breaks = [0]
        for ts in tokseqs[:-1]:
            if not ts.wrdindex_list:
                continue
            word, start = ts.wrdindex_list[-1]
            end = start + len(word)
            if end > breaks[-1]:
                breaks.append(end)
        return breaks

    def _translate_breaks_to_original_text(self, breaks, offset_map, original_text):
        """Our change: map no-punctuation break offsets back to original text."""
        translated = [0]
        for break_offset in breaks[1:]:
            if break_offset <= 0 or not offset_map:
                continue
            map_idx = min(break_offset - 1, len(offset_map) - 1)
            original_offset = offset_map[map_idx] + 1
            original_offset = self._advance_to_token_boundary(
                original_text,
                original_offset,
            )
            if original_offset > translated[-1]:
                translated.append(original_offset)
        return translated

    def _advance_to_token_boundary(self, text: str, offset: int) -> int:
        """Our change: include punctuation attached to the selected word."""
        while offset < len(text) and self._TRAILING_NONSPACE.match(text[offset]):
            offset += 1
        return offset


@chunker("text_tiling")
class TextTilingChunker(BaseChunker):
    """Wraps NLTK's TextTilingTokenizer scoring for text segmentation.

    This follows the implementation shipped in NLTK (originating from Hearst, 1997),
    except selected gaps are snapped to pseudosentence ends instead of paragraph
    breaks. Parameters are passed through from config.

    Config options (merged with configs/chunkers/text_tiling.yaml defaults):
        w (int): Token-sequence size (pseudosentence length); must be > 0.
        k (int): Number of token-sequences per block; must be > 0.
        similarity_method (str or constant): "block_comparison" or "vocabulary_introduction".
        stopwords (List[str] or None): Stopword list passed to NLTK.
        stopwords_language (str): NLTK stopwords language used when stopwords is None.
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
        self.tokenizer = PseudosentenceBoundaryTextTilingTokenizer(
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
        if value is None or value == "None":
            language = str(self.config.get("stopwords_language", "english"))
            try:
                return _load_nltk_stopwords(language)
            except (LookupError, OSError) as exc:
                if language.lower() in {"polish", "pl"}:
                    return _load_polish_stopwords()
                raise LookupError(
                    f"Stopwords for language '{language}' are not available. Run "
                    "`python -m nltk.downloader stopwords`, or set stopwords: [] "
                    "to disable stopword filtering."
                ) from exc
        return value

    def _as_int(self, key: str) -> int:
        # Validate required integer parameters from config.
        val = int(self.config[key])
        if val <= 0:
            raise ValueError(f"{key} must be positive")
        return val
