from typing import List, Dict, Any, Optional
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from ..core.base import BaseChunker, Chunk
from ..core.progress import coerce_progress_enabled, iter_with_progress
from ..core.registry import chunker
from ..core.hf_cache import resolve_hf_cache_dir

def process_sentences(sentences, embeddings, fixed_threshold=0.6, c=0.9, init_constant=1.5):
    """
    Process sentences into paragraphs based on semantic similarity. Original implementation from https://github.com/hsdslab/MaxMinChunking?tab=readme-ov-file.

    Args:
    - sentences (list of str): List of sentences to process.
    - embeddings (np.array): Sentence embeddings of shape (n_sentences, embedding_dim).
    - fixed_threshold (float): Fixed similarity threshold for joining sentences.
    - c (float): Coefficient for adjusting the similarity threshold.
    - init_constant (float): Initial constant for similarity comparison when cluster size is 1.

    Returns:
    - list of list of str: List of paragraphs, where each paragraph is a list of sentences.
    """
    
    def sigmoid(x):
        """Sigmoid function for adjusting threshold based on cluster size."""
        return 1 / (1 + np.exp(-x))

    paragraphs = []
    current_paragraph = [sentences[0]]
    cluster_start, cluster_end = 0, 1
    pairwise_min = -float('inf')

    for i in range(1, len(sentences)):
        cluster_embeddings = embeddings[cluster_start:cluster_end]

        if cluster_end - cluster_start > 1:
            new_sentence_similarities = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]

            # Adjust threshold based on cluster size and similarity
            adjusted_threshold = pairwise_min * c * sigmoid((cluster_end - cluster_start) - 1)
            new_sentence_similarity = np.max(new_sentence_similarities)
            
            # Use the minimum of the minimum similarities and the pairwise_min
            pairwise_min = min(np.min(new_sentence_similarities), pairwise_min)
        else:
            adjusted_threshold = 0
            # Use an initial constant when there's only one sentence in the cluster
            pairwise_min = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]
            new_sentence_similarity = init_constant * pairwise_min

        # Decide whether to add the sentence to the current paragraph or start a new one
        if new_sentence_similarity > max(adjusted_threshold, fixed_threshold):
            current_paragraph.append(sentences[i])
            cluster_end += 1
        else:
            paragraphs.append(current_paragraph)
            current_paragraph = [sentences[i]]
            cluster_start, cluster_end = i, i + 1
            pairwise_min = -float('inf')

    # Append the last paragraph
    paragraphs.append(current_paragraph)
    return paragraphs


@chunker("max_min_chunker")
class MaxMinChunker(BaseChunker):
    """
    Params:
        model_name: HuggingFace embedding model ID (default: 'BAAI/bge-m3')
        c_param: Relaxation parameter 'c' (default: 0.9)
        hard_threshold: Fixed similarity threshold (default: 0.6)
        init_const: Initial constant (default: 1.5)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        config = config or {}

        self.model_name = config.get("model_name", "BAAI/bge-m3") 
        self.c_param = float(config.get("c_param", 0.9))
        self.hard_threshold = float(config.get("hard_threshold", 0.6))
        self.init_const = float(config.get("init_const", 1.5))
        self._hf_cache_dir = resolve_hf_cache_dir(config)
        
        print(f"Loading MaxMin model: {self.model_name}")
        self.embedding_model = SentenceTransformer(
            self.model_name,
            cache_folder=self._hf_cache_dir,
        )

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
            iter_with_progress(documents, desc="MaxMin Chunking", enabled=show_progress)
        ):
            meta = documents_meta[idx] if documents_meta is not None else None
            all_chunks.extend(self._split_single(text, meta))

        return all_chunks

    def _split_single(
        self,
        text: str,
        document_meta: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        
        sentences = sent_tokenize(text)
        if not sentences:
            return []
        
        embeddings = self.embedding_model.encode(sentences)

        paragraphs_list = process_sentences(
            sentences=sentences,
            embeddings=embeddings,
            fixed_threshold=self.hard_threshold, 
            c=self.c_param,
            init_constant=self.init_const
        )

        chunks = []
        for i, paragraph_sentences in enumerate(paragraphs_list):
            chunk_text = " ".join(paragraph_sentences)
            new_chunk = Chunk(
                text=chunk_text,
                metadata={
                    **(document_meta or {}),
                    "chunk_index": i,
                    "method": "max_min_chunker"
                },
            )
            chunks.append(new_chunk)

        return chunks
