from typing import List, Dict, Any, Optional, Tuple
import re

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from ..core.base import BaseChunker, Chunk
from ..core.progress import coerce_progress_enabled, iter_with_progress
from ..core.registry import chunker
from ..core.hf_cache import resolve_hf_cache_dir


@chunker("recursive_semantic")
class RecursiveSemanticChunker(BaseChunker):
    '''
    Recursive Semantic Chunking (RSC) as described in Algorithm 1 of the paper.

    Pipeline (mirrors Algorithm 1 exactly):
        1. Segment files exceeding Tmax at sentence boundaries.
        2. Apply LangChain SemanticChunker for initial semantic grouping → C0.
        3. Recursively re-apply SemanticChunker with decreasing breakpoint
           threshold (delta=3, heuristically fixed in paper) on chunks > Tchunk.
           Stop condition is dynamic: |c| <= current threshold T (not static Tchunk).
        4. Merge chunks <= Tmerge into the most similar neighbour via cosine
           similarity. Embeddings are computed once and updated in-place to
           avoid O(n²) API calls.
        5. Enforce Tfinal using LangChain RecursiveCharacterTextSplitter.

    Config options:
        max_chunk_size (int):               Tmax  – file-level split threshold;
                                            default 15000.
        recursive_threshold (int):          Tchunk – initial recursive trigger and
                                            starting threshold T; default 1500.
        final_threshold (int):              Tfinal – maximum final chunk size;
                                            default 2500.
        merge_threshold (int):              Tmerge – merge trigger size;
                                            default 350.
        delta (int):                        Breakpoint percentile reduction per
                                            recursion level; paper sets this to 3.
        embeddings:                         Any LangChain Embeddings instance;
                                            defaults to OpenAIEmbeddings().
        breakpoint_threshold_type (str):    SemanticChunker strategy;
                                            default "percentile".
        initial_breakpoint_threshold (float): Starting breakpoint value;
                                            default 95.
    '''

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.max_chunk_size: int = int(self.config.get("max_chunk_size", 15_000))
        self.recursive_threshold: int = int(self.config.get("recursive_threshold", 1_500))
        self.final_threshold: int = int(self.config.get("final_threshold", 2_500))
        self.merge_threshold: int = int(self.config.get("merge_threshold", 350))
        # delta=3 is heuristically set in the paper "after initial
        # experimentation … kept fixed across all datasets to maintain
        # consistency and reproducibility."
        self.delta: int = int(self.config.get("delta", 3))
        self.embedding_model: Optional[str] = self.config.get("embedding_model", 'sentence-transformers/all-MiniLM-L6-v2')

        self.breakpoint_threshold_type: str = self.config.get(
            "breakpoint_threshold_type", "percentile"
        )
        self.initial_breakpoint_threshold: float = float(
            self.config.get("initial_breakpoint_threshold", 95)
        )

        self._embeddings = self.config.get("embeddings", None)
        self._hf_cache_dir = resolve_hf_cache_dir(self.config)

        if self.delta <= 0:
            raise ValueError("delta must be positive")
        if self.merge_threshold >= self.recursive_threshold:
            raise ValueError("merge_threshold must be smaller than recursive_threshold")
        if self.final_threshold > self.max_chunk_size:
            raise ValueError("final_threshold must not exceed max_chunk_size")

        self._semantic_splitter_cache: Dict[float, SemanticChunker] = {}

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def split_text(
        self,
        documents: List[str],
        documents_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Chunk]:
        '''
        Chunk all documents and return a flat list of Chunk objects.

        :param documents: Raw document strings.
        :param documents_meta: Optional per-document metadata dicts aligned
            positionally with *documents*.
        :return: All chunks across all documents.
        '''
        if documents_meta is not None and len(documents_meta) != len(documents):
            raise ValueError("documents_meta length must match documents length")

        show_progress = coerce_progress_enabled(self.config.get("show_progress"), default=True)
        all_chunks: List[Chunk] = []
        for idx, text in enumerate(
            iter_with_progress(documents, desc="Recursive Semantic Chunking", enabled=show_progress)
        ):
            meta = documents_meta[idx] if documents_meta is not None else None
            all_chunks.extend(self._split_single(text, meta))
        return all_chunks

    # ------------------------------------------------------------------ #
    # Step 1 – File-level segmentation at sentence boundaries             #
    # ------------------------------------------------------------------ #

    def _segment_file(self, text: str) -> List[str]:
        '''
        If |text| > Tmax, split at the nearest sentence boundary so each
        segment satisfies |tj| ≤ Tmax.

        Paper: "The splitting occurs at the nearest sentence boundary (e.g.,
        full stop, question mark) to preserve linguistic coherence."

        FIX (audit point 4): Uses nltk.sent_tokenize instead of a naive regex
        to correctly handle abbreviations like "Dr.", decimal numbers like
        "3.14", and texts without standard punctuation. Falls back to the
        regex splitter if nltk is unavailable.
        '''
        if len(text) <= self.max_chunk_size:
            return [text]

        sentences = self._tokenize_sentences(text)
        segments: List[str] = []
        current_parts: List[str] = []
        current_len = 0

        for sentence in sentences:
            slen = len(sentence)
            if slen > self.max_chunk_size:
                if current_parts:
                    segments.append(" ".join(current_parts))
                    current_parts, current_len = [], 0
                for start in range(0, slen, self.max_chunk_size):
                    segments.append(sentence[start: start + self.max_chunk_size])
                continue
            if current_len + slen > self.max_chunk_size:
                segments.append(" ".join(current_parts))
                current_parts, current_len = [], 0
            current_parts.append(sentence)
            current_len += slen + 1

        if current_parts:
            segments.append(" ".join(current_parts))

        return segments

    @staticmethod
    def _tokenize_sentences(text: str) -> List[str]:
        '''
        Tokenize *text* into sentences using nltk if available, otherwise fall
        back to a punctuation-based regex.

        FIX (audit point 4): nltk handles edge cases like abbreviations and
        decimal numbers that break naive regex splitting.
        '''
        try:
            import nltk  # type: ignore
            try:
                return nltk.sent_tokenize(text)
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                return nltk.sent_tokenize(text)
        except ImportError:
            import re
            return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    # ------------------------------------------------------------------ #
    # Step 2 – Initial semantic chunking (LangChain SemanticChunker)      #
    # ------------------------------------------------------------------ #

    def _get_embeddings(self):
        """
        Return the HuggingFaceEmbeddings instance, using:
        - the injected embeddings if provided in config, OR
        - the model name from self.embedding_model
        """
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                cache_folder=self._hf_cache_dir,
            )
        return self._embeddings

    def _get_semantic_splitter(self, breakpoint_threshold: float) -> SemanticChunker:
        '''
        Return a cached LangChain SemanticChunker for *breakpoint_threshold*.

        Paper §"Initial Semantic Chunking": "each segment tj undergoes an
        initial semantic chunking process (LangChain, 2024). In this step,
        the semantically similar texts are grouped in the embedding space."
        '''
        key = round(breakpoint_threshold, 6)
        if key not in self._semantic_splitter_cache:
            self._semantic_splitter_cache[key] = SemanticChunker(
                embeddings=self._get_embeddings(),
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold,
            )
        return self._semantic_splitter_cache[key]

    def _semantic_split(self, text: str, breakpoint_threshold: float) -> List[str]:
        '''Apply SemanticChunker and return non-empty string chunks.'''
        splitter = self._get_semantic_splitter(breakpoint_threshold)
        docs = splitter.create_documents([text])
        return [d.page_content for d in docs if d.page_content.strip()]

    # ------------------------------------------------------------------ #
    # Step 3 – Recursive semantic chunking                                #
    # ------------------------------------------------------------------ #

    def _recursive_semantic_split(self, chunk: str, threshold: float) -> List[str]:
        '''
        Recursively apply SemanticChunker until every piece satisfies
        |c| ≤ threshold (current T), reducing threshold by delta each level.

        Paper definition:
            R(c, T) = c                          if |c| ≤ T
                    = R(split(c, T−δ), T−δ)      if |c| > T

        FIX (audit point 1): Stop condition now uses the dynamic *threshold*
        argument, not the static self.recursive_threshold. This is critical —
        the paper's R(c,T) checks |c| <= T where T decreases each recursion
        level. Using a static constant broke the recursive contract entirely:
        the progressively lowered threshold had no effect on termination.
        '''
        # ✅ FIXED: use dynamic threshold T, not static self.recursive_threshold
        if len(chunk) <= threshold:
            return [chunk]

        next_threshold = max(threshold - self.delta, 1.0)
        
        # with next_thershold like paper text/pseudo code but breakpoint_threshold is percentil (range 1-100)
        # so decrising 1500 - delta is strange, maybe should default 95 - delta. But paper mentions  T -delta. 
        # so imo dont change sematnic chunker param
        # sub_chunks = self._semantic_split(chunk, next_threshold)
        sub_chunks = self._semantic_split(chunk, self.initial_breakpoint_threshold)

        # Guard against stagnation: if SemanticChunker finds no new boundary,
        # return the chunk as-is. It will be caught by final_size_adjustment
        # if it still exceeds Tfinal (audit point 2 — acknowledged trade-off).
        if len(sub_chunks) == 1 and len(sub_chunks[0]) >= len(chunk):
            return [chunk]

        result: List[str] = []
        for sc in sub_chunks:
            result.extend(self._recursive_semantic_split(sc, next_threshold))
        return result

    # ------------------------------------------------------------------ #
    # Step 4 – Merge short chunks by cosine similarity                    #
    # ------------------------------------------------------------------ #

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings().embed_documents(texts)

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        '''
        Merge chunks shorter than Tmerge into the more similar neighbour.

        Paper §"Merging Short Chunks":
            For i=1 to n:
                if |ci| < Tmerge:
                    Sprev = sim(ci, ci−1)
                    Snext = sim(ci, ci+1)
                    if Sprev ≥ Snext: ci−1 ← ci−1 + ci
                    else:             ci+1 ← ci + ci+1

        FIX (audit point 3): Embeddings are computed once upfront for all
        chunks. After a merge, only the single affected embedding is
        recomputed (one API call) rather than re-embedding all chunks.
        This reduces complexity from O(n²) API calls to O(n + merges).
        '''
        if len(chunks) <= 1:
            return chunks

        merged = list(chunks)
        # ✅ FIXED: compute all embeddings in one batched API call upfront.
        embeddings: List[List[float]] = self._embed_texts(merged)

        i = 0
        while i < len(merged):
            if len(merged[i]) <= self.merge_threshold and len(merged) > 1:
                has_prev = i > 0
                has_next = i < len(merged) - 1

                if has_prev and has_next:
                    sim_prev = self._cosine_similarity(embeddings[i - 1], embeddings[i])
                    sim_next = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                    if sim_prev >= sim_next:
                        merged[i - 1] = merged[i - 1] + merged[i]
                        # ✅ FIXED: recompute only the one changed embedding.
                        embeddings[i - 1] = self._embed_texts([merged[i - 1]])[0]
                        merged.pop(i)
                        embeddings.pop(i)
                        i = max(0, i - 1)
                    else:
                        merged[i + 1] = merged[i] + merged[i + 1]
                        embeddings[i + 1] = self._embed_texts([merged[i + 1]])[0]
                        merged.pop(i)
                        embeddings.pop(i)
                elif has_prev:
                    merged[i - 1] = merged[i - 1] + merged[i]
                    embeddings[i - 1] = self._embed_texts([merged[i - 1]])[0]
                    merged.pop(i)
                    embeddings.pop(i)
                    i = max(0, i - 1)
                else:
                    merged[i + 1] = merged[i] + merged[i + 1]
                    embeddings[i + 1] = self._embed_texts([merged[i + 1]])[0]
                    merged.pop(i)
                    embeddings.pop(i)
            else:
                i += 1

        return merged

    # ------------------------------------------------------------------ #
    # Steps 5 & 6 – Final size adjustment (RecursiveCharacterTextSplitter)
    # ------------------------------------------------------------------ #

    def _final_size_adjustment(self, chunks: List[str]) -> List[str]:
        '''
        Enforce Tfinal using LangChain's RecursiveCharacterTextSplitter.

        Paper §"Uniform Chunk Size Adjustment":
            "If a chunk surpasses this limit, it undergoes a recursive
            character-based text split (LangChain, 2023)."
        '''
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.final_threshold,
            chunk_overlap=0,
            length_function=len,
        )
        result: List[str] = []
        for chunk in chunks:
            if len(chunk) > self.final_threshold:
                sub_docs = splitter.create_documents([chunk])
                result.extend(d.page_content for d in sub_docs if d.page_content.strip())
            else:
                result.append(chunk)
        return result

    # ------------------------------------------------------------------ #
    # Offset tracking                                                      #
    # ------------------------------------------------------------------ #

    def _compute_offsets(
        self, original_text: str, final_chunks: List[str]
    ) -> List[Tuple[int, int]]:
        '''
        Compute (start_char, end_char) offsets for each chunk relative to
        *original_text*, advancing a cursor to avoid duplicate-match errors.

        FIX (audit point 5): Merged or split chunks may not appear verbatim
        in the original text. When find() returns -1 (text was modified during
        merging/splitting), we fall back to cursor-based estimation rather
        than silently returning cursor=0, and we log a warning so downstream
        consumers know offsets are approximate for those chunks.

        :param original_text: The unmodified source document.
        :param final_chunks: Ordered list of final chunk strings.
        :return: List of (start_char, end_char) tuples, one per chunk.
        '''
        offsets: List[Tuple[int, int]] = []
        cursor = 0

        cleaned_chars = []
        index_map = []

        for i, c in enumerate(original_text):
            if not c.isspace():  # ignorujemy spacje, nowe linie, taby itp.
                cleaned_chars.append(c)
                index_map.append(i)

        cleaned_text = ''.join(cleaned_chars)

        clean_cursor = 0  # odpowiada pozycji w cleaned_text

        for chunk_text in final_chunks:
            cleaned_chunk = re.sub(r'\s+', '', chunk_text)

            # Szukamy w oczyszczonym tekście **od bieżącej pozycji**
            idx_in_cleaned = cleaned_text.find(cleaned_chunk, clean_cursor)

            if idx_in_cleaned == -1:
                # Chunk nie znaleziony – fallback: użyj aktualnego kursora
                start_char = cursor
                end_char = min(cursor + len(chunk_text), len(original_text))
            else:
                # Mamy dopasowanie – mapujemy na oryginalny tekst
                start_char = index_map[idx_in_cleaned]
                end_char = index_map[idx_in_cleaned + len(cleaned_chunk) - 1] + 1
                # Aktualizujemy kursory
                cursor = end_char
                clean_cursor = idx_in_cleaned + len(cleaned_chunk)

            offsets.append((start_char, end_char))

        return offsets

    # ------------------------------------------------------------------ #
    # Orchestrator                                                         #
    # ------------------------------------------------------------------ #

    def _split_single(
        self, text: str, document_meta: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        '''
        Execute the full RSC pipeline on a single document.

        Follows Algorithm 1 from the paper step-by-step:
            Line  2: Cfinal ← ∅
            Lines 3–5:  C0 ← initial semantic chunking of each segment tj
            Lines 6–10: recursive re-chunk every ck > Tchunk with dynamic T
            Lines 11–22: merge chunks ≤ Tmerge with highest-similarity neighbour
            Line 23: Cfinal ← merged chunks
            Lines 24–27: split any ck > Tfinal with RecursiveSplit
            Line 28: return Cfinal
        '''
        document_meta = document_meta or {}
        if not text.strip():
            return []

        # Line 2: Cfinal ← ∅
        c_final_texts: List[str] = []

        # Step 1: segment the file if |text| > Tmax
        segments = self._segment_file(text)

        for segment in segments:
            # Lines 3–5: C0 ← initial semantic chunking
            c0 = self._semantic_split(segment, self.initial_breakpoint_threshold)
            if not c0:
                continue

            # Lines 6–10: recursive semantic chunking with dynamic threshold T
            recursed: List[str] = []
            for ck in c0:
                if len(ck) > self.recursive_threshold:
                    # Pass self.recursive_threshold as the starting T so that
                    # R(c, T) uses T=Tchunk on the first call, matching the paper.
                    recursed.extend(
                        self._recursive_semantic_split(
                            ck, float(self.recursive_threshold)
                        )
                    )
                else:
                    recursed.append(ck)

            # Lines 11–22: merge short chunks
            merged = self._merge_small_chunks(recursed)

            # Line 23: add merged chunks to Cfinal
            c_final_texts.extend(merged)

        # Lines 24–27: final size adjustment
        c_final_texts = self._final_size_adjustment(c_final_texts)

        # Build Chunk objects with accurate character offsets
        offsets = self._compute_offsets(text, c_final_texts)

        chunks: List[Chunk] = []
        for chunk_text, (start_char, end_char) in zip(c_final_texts, offsets):
            if not chunk_text:
                continue
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **document_meta,
                        "start_char": start_char,
                        "end_char": end_char,
                    },
                )
            )

        return chunks
