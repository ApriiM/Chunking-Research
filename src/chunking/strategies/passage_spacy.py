from typing import Any, Dict, List, Optional

from ..core.base import BaseChunker, Chunk
from ..core.registry import chunker


@chunker("passage_spacy")
class SpacyPassageChunker(BaseChunker):
    """
    Groups text into passages of N sentences using spaCy sentence segmentation.

    Config options (merged with configs/chunkers/passage_spacy.yaml defaults):
        passage_length (int): Sentences per chunk; must be > 0. Setting passage_length=1 yields
            one sentence per chunk (sentence-level segmentation).
        spacy_model (str): spaCy model to load (e.g., "en_core_web_sm").
        use_sentencizer (bool): Add the sentencizer component if no parser is available.
        disable (List[str]): Pipeline components to disable for speed (e.g., ["ner", "textcat"]).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SpacyPassageChunker with spaCy model and segmentation options.

        :param config: Configuration dictionary with:
            - passage_length (int, >0)
            - spacy_model (str)
            - use_sentencizer (bool)
            - disable (List[str])
        :type config: Optional[Dict[str, Any]]
        """
        super().__init__(config)
        self.passage_length: int = int(self.config["passage_length"])
        if self.passage_length <= 0:
            raise ValueError("passage_length must be positive")

        self.model: str = str(self.config.get("spacy_model", "en_core_web_sm"))
        self.use_sentencizer: bool = bool(self.config.get("use_sentencizer", True))
        self.disable = list(self.config.get("disable", ["ner", "textcat"]))
        self._nlp = None

    def _load_nlp(self):
        if self._nlp is not None:
            return
        try:
            import spacy
        except ImportError as exc:
            raise ImportError(
                "spaCy is required for 'passage_spacy'. Install with `pip install spacy` and a model, e.g. `python -m spacy download en_core_web_sm`."
            ) from exc

        try:
            # Prefer the requested model if installed.
            nlp = spacy.load(self.model, disable=self.disable)
        except OSError as exc:
            raise OSError(
                f"spaCy model '{self.model}' not found. Install it with "
                f"`python -m spacy download {self.model}` or set spacy_model to an installed model."
            ) from exc
        else:
            # Ensure sentence boundaries exist if parser is disabled.
            if self.use_sentencizer and "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        self._nlp = nlp

    def split_text(
        self,
        documents: List[str],
        documents_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Chunk]:
        # Ensure per-document metadata aligns with the input list.
        if documents_meta is not None and len(documents_meta) != len(documents):
            raise ValueError("documents_meta length must match documents length")

        # Lazy-load spaCy pipeline only when chunking is invoked.
        self._load_nlp()

        all_chunks: List[Chunk] = []
        for idx, text in enumerate(documents):
            meta = documents_meta[idx] if documents_meta is not None else None
            all_chunks.extend(self._split_single(text, meta))
        return all_chunks

    def _split_single(
        self, text: str, document_meta: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        document_meta = document_meta or {}
        if not text:
            return []

        # Run spaCy, then extract sentence spans.
        doc = self._nlp(text)
        sentences = [sent for sent in doc.sents if sent.text.strip()]
        if not sentences:
            return []

        chunks: List[Chunk] = []
        for start_idx in range(0, len(sentences), self.passage_length):
            group = sentences[start_idx : start_idx + self.passage_length]
            if not group:
                continue
            # Join sentence texts and keep character offsets from the original doc.
            chunk_text = " ".join(sent.text.strip() for sent in group if sent.text.strip())
            if not chunk_text:
                continue
            end_idx = start_idx + len(group)
            start_char = group[0].start_char
            end_char = group[-1].end_char
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **document_meta,
                        "start_sentence": start_idx,
                        "end_sentence": end_idx,
                        "start_char": start_char,
                        "end_char": end_char,
                    },
                )
            )
        return chunks
