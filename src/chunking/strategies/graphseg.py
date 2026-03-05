# GraphSeg resources (download once):
#   mkdir -p data/graphseg/embeddings data/graphseg/frequencies data/graphseg/resources
#   curl -L -o data/graphseg/embeddings/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
#   curl -L -o data/graphseg/frequencies/freqs.txt https://bitbucket.org/gg42554/graphseg/raw/c551cfa0926ed3990cefaaf44997a5ce48ff3e84/source/res/freqs.txt
#   curl -L -o data/graphseg/resources/stopwords.txt https://bitbucket.org/gg42554/graphseg/raw/c551cfa0926ed3990cefaaf44997a5ce48ff3e84/source/res/stopwords.txt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.base import BaseChunker, Chunk
from ..core.progress import coerce_progress_enabled, iter_with_progress
from ..core.registry import chunker


def _ensure_graphseg_submodule() -> None:
    root = Path(__file__).resolve().parents[3]
    submodule_root = root / "submodules" / "graphseg_python"
    if not submodule_root.exists():
        raise FileNotFoundError(
            f"graphseg_python submodule not found at {submodule_root}. "
            "Run: git submodule update --init --recursive"
        )
    import sys

    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))


@chunker("graphseg")
class GraphSegChunker(BaseChunker):
    """
    Wrapper around external GraphSeg implementation (submodules/graphseg_python).

    Config options:
        tau (float): edge threshold (maps to GraphSeg.treshold).
        min_segment_size (int): minimum segment size (maps to GraphSeg.minseg).
        embedding_path (str): path to GloVe zip or text file (GraphSeg path_wordvecs).
        frequency_path (str): path to frequency file (GraphSeg path_freqs).
        stopwords_path (str): path to stopwords file (GraphSeg path_stop).
        spacy_model (str): spaCy model name (default: en_core_web_sm).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.tau: float = float(self.config.get("tau", 0.25))
        self.min_segment_size: int = int(self.config.get("min_segment_size", 3))
        self.embedding_path: Optional[str] = self.config.get("embedding_path")
        self.frequency_path: Optional[str] = self.config.get("frequency_path")
        self.stopwords_path: Optional[str] = self.config.get("stopwords_path")
        self.spacy_model: str = str(self.config.get("spacy_model", "en_core_web_sm"))

        if not self.embedding_path or not self.frequency_path or not self.stopwords_path:
            raise ValueError("graphseg requires embedding_path, frequency_path, stopwords_path")

        _ensure_graphseg_submodule()

        try:
            from graphseg.graphseg import GraphSeg
        except Exception as exc:
            raise ImportError(
                "Failed to import graphseg-python. Ensure its dependencies are installed "
                "(genutility, spacy, scipy, networkx)."
            ) from exc

        try:
            import spacy
        except Exception as exc:
            raise ImportError(
                "spaCy is required for graphseg. Install spacy and an English model."
            ) from exc

        self._graphseg = GraphSeg(
            self.frequency_path,
            self.embedding_path,
            self.stopwords_path,
            self.tau,
            self.min_segment_size,
        )
        self._nlp = spacy.load(self.spacy_model)

    def split_text(
        self,
        documents: List[str],
        documents_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Chunk]:
        if documents_meta is not None and len(documents_meta) != len(documents):
            raise ValueError("documents_meta length must match documents length")

        show_progress = coerce_progress_enabled(self.config.get("show_progress"), default=True)
        all_chunks: List[Chunk] = []
        for idx, text in enumerate(
            iter_with_progress(documents, desc="GraphSeg Chunking", enabled=show_progress)
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

        doc = self._nlp(text)
        sentences = list(doc.sents)
        index_by_span: Dict[Tuple[int, int], int] = {
            (sent.start_char, sent.end_char): idx for idx, sent in enumerate(sentences)
        }

        chunks: List[Chunk] = []
        for seg_idx, seg in enumerate(self._graphseg.segment(doc)):
            if not seg:
                continue
            start_char = seg[0].start_char
            end_char = seg[-1].end_char
            chunk_text = doc.text[start_char:end_char]
            if not chunk_text.strip():
                continue

            start_idx = index_by_span.get((seg[0].start_char, seg[0].end_char))
            end_idx = index_by_span.get((seg[-1].start_char, seg[-1].end_char))
            end_idx = (end_idx + 1) if end_idx is not None else None

            metadata = {
                **document_meta,
                "segment_index": seg_idx,
                "start_char": start_char,
                "end_char": end_char,
            }
            if start_idx is not None:
                metadata["start_sentence"] = start_idx
            if end_idx is not None:
                metadata["end_sentence"] = end_idx

            chunks.append(Chunk(text=chunk_text, metadata=metadata))

        return chunks
