from typing import List, Dict, Any, Optional
import numpy as np
from scipy.sparse import diags
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from ..core.base import BaseChunker, Chunk
from ..core.registry import chunker

@chunker("sequential_hac_chunker")
class SequentialHACChunker(BaseChunker):
    """
    Implementation of the Sequential Hierarchical Agglomerative Clustering (HAC) for RAG.
    
    This method uses Single-Linkage Agglomerative Clustering with a strict connectivity constraint.
    It ensures that only spatially adjacent sentences in the text can be merged into a single chunk.

    Params:
        model_name (str): HuggingFace embedding model ID (default: 'BAAI/bge-m3').
        distance_threshold (float): The linkage distance threshold at or above which clusters 
                                    will not be merged. Lower value = more chunks. (default: 0.3).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        config = config or {}

        self.model_name = config.get("model_name", "BAAI/bge-m3") 
        self.distance_threshold = float(config.get("distance_threshold", 0.3))
        
        print(f"Loading Sequential HAC model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)

    def split_text(
        self,
        documents: List[str],
        documents_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Chunk]:
    
        if documents_meta is not None and len(documents_meta) != len(documents):
            raise ValueError("documents_meta length must match documents length")

        all_chunks: List[Chunk] = []
        
        for idx, text in enumerate(tqdm(documents, desc="Sequential HAC Chunking")):
            meta = documents_meta[idx] if documents_meta is not None else None
            all_chunks.extend(self._split_single(text, meta))

        return all_chunks

    def _split_single(
        self,
        text: str,
        document_meta: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        
        # Basic Segmentation
        sentences = sent_tokenize(text)
        n_sentences = len(sentences)

        if n_sentences == 0:
            return []
        if n_sentences == 1:
            return [
                Chunk(
                    text=sentences[0], 
                    metadata={
                        **(document_meta or {}), 
                        "chunk_index": 0, 
                        "method": "sequential_hac_chunker"
                    }
                )
            ]
        
        # Generate Embeddings
        embeddings = self.embedding_model.encode(
            sentences, 
            batch_size=32, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )

        # Define Connectivity Constraint
        # Create a sparse matrix with 1s on the first diagonals above and below the main diagonal.
        # This restricts merges to strictly adjacent sentences (e.g., sentence i can only merge with i-1 or i+1).
        connectivity = diags([1, 1], [-1, 1], shape=(n_sentences, n_sentences), dtype=int)

        # Initialize and Run Agglomerative Clustering
        model = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=self.distance_threshold,
            metric='cosine', 
            linkage='single',
            connectivity=connectivity
        )

        labels = model.fit_predict(embeddings)

        # Group Sentences into Chunks
        # Because of the connectivity constraint, identical labels will form contiguous blocks.
        # We iterate through the labels and group sentences when the label remains the same.
        
        chunks = []
        current_chunk_sentences = []
        current_label = labels[0]
        chunk_idx = 0

        for i, label in enumerate(labels):
            if label == current_label:
                current_chunk_sentences.append(sentences[i])
            else:
                # Save the accumulated sentences as a completed chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            **(document_meta or {}),
                            "chunk_index": chunk_idx,
                            "method": "sequential_hac_chunker"
                        }
                    )
                )
                
                # Reset for the new chunk
                chunk_idx += 1
                current_chunk_sentences = [sentences[i]]
                current_label = label
        
        # Append the final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **(document_meta or {}),
                        "chunk_index": chunk_idx,
                        "method": "sequential_hac_chunker"
                    }
                )
            )

        return chunks