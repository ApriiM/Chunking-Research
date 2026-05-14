import re
from bisect import bisect_left
from collections import Counter
from typing import Dict, Any, List, Optional

# CLI examples:
# Original internals, sentence-boundary normalization:
#   python -m src.chunking.prepare_passages --documents-path data/processed/qasper/test/documents/documents.jsonl --chunker-name text_tiling --chunker-params "{sentence_language: english, stopwords: null, stopwords_language: english, use_block_comparison_v2: false, use_depth_scores_v2: false, use_normalize_boundaries_v2: false}" --output-path data/processed/qasper/test/passages_all/passages_text_tiling_sentence_en_original_internals.jsonl --overwrite
# Fast internals, same scoring math:
#   python -m src.chunking.prepare_passages --documents-path data/processed/qasper/test/documents/documents.jsonl --chunker-name text_tiling --chunker-params "{sentence_language: english, stopwords: null, stopwords_language: english, use_block_comparison_v2: true, use_depth_scores_v2: true, use_normalize_boundaries_v2: true}" --output-path data/processed/qasper/test/passages_all/passages_text_tiling_sentence_en_v2_internals.jsonl --overwrite

import numpy
import math
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


_SPACY_SENTENCIZERS = {}


def _spacy_language_code(language: str) -> str:
    normalized = (language or "english").strip().lower()
    mapping = {
        "english": "en",
        "eng": "en",
        "en": "en",
        "polish": "pl",
        "polski": "pl",
        "pl": "pl",
    }
    return mapping.get(normalized, normalized)


def _get_spacy_sentencizer(language: str):
    code = _spacy_language_code(language)
    if code not in _SPACY_SENTENCIZERS:
        import spacy

        nlp = spacy.blank(code)
        nlp.add_pipe("sentencizer")
        _SPACY_SENTENCIZERS[code] = nlp
    return _SPACY_SENTENCIZERS[code]


class SentenceBoundaryTextTilingTokenizer(TextTilingTokenizer):
    """NLTK TextTiling with final boundaries snapped to sentence ends.

    NLTK's original implementation scores lexical cohesion at pseudosentence
    gaps, then snaps selected gaps to paragraph breaks. This variant keeps the
    scoring pipeline unchanged and replaces only the final legal boundary
    positions: selected gaps are normalized to nearest spaCy sentence end.
    """

    _NOPUNCT_ALLOWED = re.compile(r"[a-z\-' \n\t]")

    def __init__(
        self,
        *args,
        sentence_language: str = "english",
        show_gap_progress: bool = False,
        use_block_comparison_v2: bool = True,
        use_depth_scores_v2: bool = True,
        use_normalize_boundaries_v2: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sentence_language = sentence_language
        self.show_gap_progress = bool(show_gap_progress)
        self.use_block_comparison_v2 = bool(use_block_comparison_v2)
        self.use_depth_scores_v2 = bool(use_depth_scores_v2)
        self.use_normalize_boundaries_v2 = bool(use_normalize_boundaries_v2)

    def tokenize(self, text):
        """Return TextTiling segments using sentence-end boundary candidates."""

        # Original NLTK step: lowercase, remove punctuation, then divide into
        # fixed-size token sequences ("pseudosentences").
        # Our change: final legal boundaries come from spaCy sentence ends
        # instead of paragraph breaks.
        lowercase_text = text.lower()
        sentence_breaks = self._mark_sentence_breaks(text)
        if len(sentence_breaks) < 2:
            return [text]
        text_length = len(lowercase_text)
        nopunct_text = "".join(
            c for c in lowercase_text if self._NOPUNCT_ALLOWED.match(c)
        )

        tokseqs = self._divide_to_tokensequences(nopunct_text)
        if len(tokseqs) < 2:
            return [text]

        # Original NLTK creates a paragraph table over punctuation-stripped
        # text. Block-comparison scoring uses token-sequence occurrences, not
        # paragraph counts, so a whole-document pseudo paragraph preserves the
        # scoring path while avoiding paragraph-format failures.
        nopunct_breaks = [0, len(nopunct_text)]

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
        # Our change: the legal offsets are sentence ends, not paragraph
        # breaks, so paragraph formatting no longer controls chunk boundaries.
        normalized_boundaries = self._normalize_boundaries(
            text,
            segment_boundaries,
            sentence_breaks,
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

    def _mark_sentence_breaks(self, text: str) -> List[int]:
        """Our change: use spaCy sentencizer sentence ends as legal breaks."""
        nlp = _get_spacy_sentencizer(self.sentence_language)
        if len(text) + 1 > nlp.max_length:
            nlp.max_length = len(text) + 1
        doc = nlp(text)
        breaks = [0]
        for sent in doc.sents:
            if sent.end_char > breaks[-1]:
                breaks.append(sent.end_char)
        return breaks

    def _identify_boundaries(self, depth_scores):
        """Original NLTK cutoff, without suppressing nearby boundaries."""

        boundaries = [0 for _ in depth_scores]

        avg = sum(depth_scores) / len(depth_scores)
        stdev = numpy.std(depth_scores)

        if self.cutoff_policy == LC:
            cutoff = avg - stdev
        else:
            cutoff = avg - stdev / 2.0

        depth_tuples = sorted(zip(depth_scores, range(len(depth_scores))))
        depth_tuples.reverse()
        hp = list(filter(lambda x: x[0] > cutoff, depth_tuples))

        for _, boundary_index in hp:
            boundaries[boundary_index] = 1

        return boundaries

    def _depth_scores(self, scores):
        """Depth scores using the configured implementation."""

        if self.use_depth_scores_v2:
            return self._depth_scores_v2(scores)
        return self._depth_scores_original(scores)

    def _depth_scores_original(self, scores):
        """Original NLTK depth scoring with optional progress."""

        depth_scores = [0 for _ in scores]
        clip = min(max(len(scores) // 10, 2), 5)
        indexed_scores = enumerate(scores[clip:-clip], start=clip)

        if self.show_gap_progress and len(scores) > 0:
            try:
                from tqdm import tqdm

                indexed_scores = tqdm(
                    indexed_scores,
                    total=max(0, len(scores) - 2 * clip),
                    desc="TextTiling depth scores",
                    unit="gap",
                )
            except ImportError:
                pass

        for index, gapscore in indexed_scores:
            lpeak = gapscore
            for score in scores[index::-1]:
                if score >= lpeak:
                    lpeak = score
                else:
                    break
            rpeak = gapscore
            for score in scores[index:]:
                if score >= rpeak:
                    rpeak = score
                else:
                    break
            depth_scores[index] = lpeak + rpeak - 2 * gapscore

        return depth_scores

    def _depth_scores_v2(self, scores):
        """Equivalent depth scoring with precomputed local monotonic peaks."""

        depth_scores = [0 for _ in scores]
        if not scores:
            return depth_scores

        clip = min(max(len(scores) // 10, 2), 5)
        if len(scores) <= 2 * clip:
            return depth_scores

        left_peaks = self._left_monotonic_peaks(scores)
        right_peaks = self._right_monotonic_peaks(scores)
        index_range = range(clip, len(scores) - clip)

        if self.show_gap_progress:
            try:
                from tqdm import tqdm

                index_range = tqdm(
                    index_range,
                    total=max(0, len(scores) - 2 * clip),
                    desc="TextTiling depth scores v2",
                    unit="gap",
                )
            except ImportError:
                pass

        for index in index_range:
            depth_scores[index] = (
                left_peaks[index] + right_peaks[index] - 2 * scores[index]
            )

        return depth_scores

    def _left_monotonic_peaks(self, scores):
        peaks = [0 for _ in scores]
        peaks[0] = scores[0]
        for index in range(1, len(scores)):
            if scores[index - 1] >= scores[index]:
                peaks[index] = peaks[index - 1]
            else:
                peaks[index] = scores[index]
        return peaks

    def _right_monotonic_peaks(self, scores):
        peaks = [0 for _ in scores]
        peaks[-1] = scores[-1]
        for index in range(len(scores) - 2, -1, -1):
            if scores[index + 1] >= scores[index]:
                peaks[index] = peaks[index + 1]
            else:
                peaks[index] = scores[index]
        return peaks

    def _normalize_boundaries(self, text, boundaries, paragraph_breaks):
        """Normalize selected gap boundaries using the configured implementation."""

        if not self.use_normalize_boundaries_v2:
            return self._normalize_boundaries_original(text, boundaries, paragraph_breaks)

        if not boundaries or not paragraph_breaks:
            return []

        normalized_boundaries = []
        char_count, word_count, gaps_seen = 0, 0, 0
        seen_word = False
        chars = text

        if self.show_gap_progress and len(text) > 0:
            try:
                from tqdm import tqdm

                chars = tqdm(
                    text,
                    total=len(text),
                    desc="TextTiling normalize boundaries",
                    unit="char",
                    mininterval=1.0,
                )
            except ImportError:
                pass

        for char in chars:
            char_count += 1
            if char in " \t\n" and seen_word:
                seen_word = False
                word_count += 1
            if char not in " \t\n" and not seen_word:
                seen_word = True
            if gaps_seen < len(boundaries) and word_count > (
                max(gaps_seen * self.w, self.w)
            ):
                if boundaries[gaps_seen] == 1:
                    bestbr = self._nearest_break(paragraph_breaks, char_count)
                    if bestbr not in normalized_boundaries:
                        normalized_boundaries.append(bestbr)
                gaps_seen += 1

        return normalized_boundaries

    def _normalize_boundaries_original(self, text, boundaries, paragraph_breaks):
        return super()._normalize_boundaries(text, boundaries, paragraph_breaks)

    def _nearest_break(self, breaks, char_count):
        insertion_index = bisect_left(breaks, char_count)
        if insertion_index <= 0:
            return breaks[0]
        if insertion_index >= len(breaks):
            return breaks[-1]

        left = breaks[insertion_index - 1]
        right = breaks[insertion_index]
        if char_count - left <= right - char_count:
            return left
        return right

    def _block_comparison(self, tokseqs, token_table):
        """Block comparison using the configured implementation."""

        if self.use_block_comparison_v2:
            return self._block_comparison_v2(tokseqs, token_table)
        return self._block_comparison_original(tokseqs, token_table)

    def _block_comparison_original(self, tokseqs, token_table):
        """Original NLTK block comparison with optional per-gap progress."""

        def blk_frq(tok, block):
            ts_occs = filter(lambda o: o[0] in block, token_table[tok].ts_occurences)
            freq = sum(tsocc[1] for tsocc in ts_occs)
            return freq

        gap_scores = []
        numgaps = len(tokseqs) - 1
        gap_range = range(numgaps)

        if self.show_gap_progress and numgaps > 0:
            try:
                from tqdm import tqdm

                gap_range = tqdm(
                    gap_range,
                    total=numgaps,
                    desc="TextTiling block comparison",
                    unit="gap",
                )
            except ImportError:
                pass

        for curr_gap in gap_range:
            score_dividend, score_divisor_b1, score_divisor_b2 = 0.0, 0.0, 0.0
            score = 0.0
            if curr_gap < self.k - 1:
                window_size = curr_gap + 1
            elif curr_gap > numgaps - self.k:
                window_size = numgaps - curr_gap
            else:
                window_size = self.k

            b1 = [ts.index for ts in tokseqs[curr_gap - window_size + 1 : curr_gap + 1]]
            b2 = [ts.index for ts in tokseqs[curr_gap + 1 : curr_gap + window_size + 1]]

            for t in token_table:
                score_dividend += blk_frq(t, b1) * blk_frq(t, b2)
                score_divisor_b1 += blk_frq(t, b1) ** 2
                score_divisor_b2 += blk_frq(t, b2) ** 2
            try:
                score = score_dividend / math.sqrt(score_divisor_b1 * score_divisor_b2)
            except ZeroDivisionError:
                pass

            gap_scores.append(score)

        return gap_scores

    def _block_comparison_v2(self, tokseqs, token_table):
        """Equivalent block comparison using sparse local word counters."""

        del token_table  # Kept in the signature to match NLTK's method contract.

        gap_scores = []
        numgaps = len(tokseqs) - 1
        gap_range = range(numgaps)

        if self.show_gap_progress and numgaps > 0:
            try:
                from tqdm import tqdm

                gap_range = tqdm(
                    gap_range,
                    total=numgaps,
                    desc="TextTiling block comparison v2",
                    unit="gap",
                )
            except ImportError:
                pass

        for curr_gap in gap_range:
            if curr_gap < self.k - 1:
                window_size = curr_gap + 1
            elif curr_gap > numgaps - self.k:
                window_size = numgaps - curr_gap
            else:
                window_size = self.k

            left_block = tokseqs[curr_gap - window_size + 1 : curr_gap + 1]
            right_block = tokseqs[curr_gap + 1 : curr_gap + window_size + 1]
            left_counts = self._block_word_counts(left_block)
            right_counts = self._block_word_counts(right_block)

            left_norm = sum(value * value for value in left_counts.values())
            right_norm = sum(value * value for value in right_counts.values())
            if not left_norm or not right_norm:
                gap_scores.append(0.0)
                continue

            if len(left_counts) <= len(right_counts):
                score_dividend = sum(
                    value * right_counts.get(token, 0)
                    for token, value in left_counts.items()
                )
            else:
                score_dividend = sum(
                    left_counts.get(token, 0) * value
                    for token, value in right_counts.items()
                )

            gap_scores.append(score_dividend / math.sqrt(left_norm * right_norm))

        return gap_scores

    def _block_word_counts(self, block):
        counts = Counter()
        for ts in block:
            counts.update(word for word, _ in ts.wrdindex_list)
        return counts


@chunker("text_tiling")
class TextTilingChunker(BaseChunker):
    """Wraps NLTK's TextTilingTokenizer scoring for text segmentation.

    This follows the implementation shipped in NLTK (originating from Hearst, 1997),
    except selected gaps are snapped to sentence ends instead of paragraph
    breaks. Parameters are passed through from config.

    Config options (merged with configs/chunkers/text_tiling.yaml defaults):
        w (int): Token-sequence size (pseudosentence length); must be > 0.
        k (int): Number of token-sequences per block; must be > 0.
        similarity_method (str or constant): "block_comparison" or "vocabulary_introduction".
        stopwords (List[str] or None): Stopword list passed to NLTK.
        stopwords_language (str): NLTK stopwords language used when stopwords is None.
        sentence_language (str): spaCy sentencizer language for final boundaries.
        use_block_comparison_v2 (bool): Use sparse block-comparison implementation.
        use_depth_scores_v2 (bool): Use linear depth-score implementation.
        use_normalize_boundaries_v2 (bool): Use binary-search boundary normalization.
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
        self.tokenizer = SentenceBoundaryTextTilingTokenizer(
            w=self._as_int("w"),
            k=self._as_int("k"),
            similarity_method=self._resolve_similarity(self.config.get("similarity_method")),
            stopwords=self._resolve_stopwords(self.config.get("stopwords")),
            smoothing_method=self._resolve_smoothing(self.config.get("smoothing_method")),
            smoothing_width=self._as_int("smoothing_width"),
            smoothing_rounds=self._as_int("smoothing_rounds"),
            cutoff_policy=self._resolve_cutoff(self.config.get("cutoff_policy")),
            demo_mode=bool(self.config.get("demo_mode", False)),
            sentence_language=self._resolve_sentence_language(),
            show_gap_progress=self._resolve_gap_progress(),
            use_block_comparison_v2=self._resolve_bool(
                "use_block_comparison_v2",
                default=True,
            ),
            use_depth_scores_v2=self._resolve_bool(
                "use_depth_scores_v2",
                default=True,
            ),
            use_normalize_boundaries_v2=self._resolve_bool(
                "use_normalize_boundaries_v2",
                default=True,
            ),
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

    def _resolve_sentence_language(self) -> str:
        return str(
            self.config.get(
                "sentence_language",
                self.config.get("stopwords_language", "english"),
            )
        )

    def _resolve_gap_progress(self) -> bool:
        return coerce_progress_enabled(
            self.config.get("show_gap_progress", self.config.get("show_progress")),
            default=True,
        )

    def _as_int(self, key: str) -> int:
        # Validate required integer parameters from config.
        val = int(self.config[key])
        if val <= 0:
            raise ValueError(f"{key} must be positive")
        return val

    def _resolve_bool(self, key: str, *, default: bool) -> bool:
        return coerce_progress_enabled(self.config.get(key), default=default)
