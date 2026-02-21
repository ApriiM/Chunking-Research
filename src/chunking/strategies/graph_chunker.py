from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
import networkx as nx
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from ..core.base import BaseChunker, Chunk
from ..core.registry import chunker

@chunker("graph_chunker")
class GraphChunker(BaseChunker):
    """
    Implementation of the Graph-Based Clique Clustering method for RAG.
    
    This fits the 'Bottom-Up' approach described in:
    "Enhancing Retrieval Augmented Generation with Hierarchical Text Segmentation Chunking"
    
    Workflow:
    1.  Input text is split into sentences (replacing the LSTM segmentation phase).
    2.  A similarity graph is constructed where nodes are sentences.
    3.  Edges are created if similarity > dynamic threshold (mean + k * std).
    4.  Maximal Cliques are detected in the graph.
    5.  ADJACENT sentences are merged if they share a common clique.
    6.  Orphan sentences (singletons) are merged into nearest neighbors.

    Params:
        model_name (str): HuggingFace embedding model ID (default: 'BAAI/bge-m3').
        k_param (float): Parameter 'k' controlling the threshold sensitivity (tau = mu + k * sigma).
                         Higher k = stricter merging, more chunks. (default: 1.0).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        config = config or {}

        self.model_name = config.get("model_name", "BAAI/bge-m3") 
        self.k_param = float(config.get("k_param", 1.0))
        
        print(f"Loading model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)

    def split_text(
        self,
        documents: List[str],
        documents_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Chunk]:
    
        if documents_meta is not None and len(documents_meta) != len(documents):
            raise ValueError("documents_meta length must match documents length")

        all_chunks: List[Chunk] = []
        
        for idx, text in enumerate(tqdm(documents, desc="Graph Chunking")):
            meta = documents_meta[idx] if documents_meta is not None else None
            all_chunks.extend(self._split_single(text, meta))

        return all_chunks

    def _split_single(
        self,
        text: str,
        document_meta: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        
        # Basic Segmentation (Sentences as nodes)
        sentences = sent_tokenize(text)
        if not sentences:
            return []
        if len(sentences) == 1:
             return [Chunk(text=sentences[0], metadata={**(document_meta or {}), "chunk_index": 0})]
        
        embeddings = self.embedding_model.encode(sentences, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

        # Building Graph and Detecting Cliques
   
        sim_matrix = cosine_similarity(embeddings)
        
        # Calculate dynamic threshold: tau = mu + k * sigma
        upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
        similarities = sim_matrix[upper_tri_indices]
        
        if len(similarities) == 0:
            mu, sigma = 0, 0
        else:
            mu = np.mean(similarities)
            sigma = np.std(similarities)
            
        threshold = mu + self.k_param * sigma

        # Construct the Graph G = (V, E)
        G = nx.Graph()
        G.add_nodes_from(range(len(sentences)))
        
        # Add edges where similarity > threshold
        rows, cols = np.where(np.triu(sim_matrix, k=1) > threshold)
        edges = zip(rows, cols)
        G.add_edges_from(edges)

        # Finding Maximal Cliques
    
        cliques = list(nx.find_cliques(G))
        
        # Map each sentence index to the list of cliques it belongs to
        sentence_to_cliques: Dict[int, Set[int]] = {i: set() for i in range(len(sentences))}
        for clique_id, clique_nodes in enumerate(cliques):
            for node in clique_nodes:
                sentence_to_cliques[node].add(clique_id)

        # Merging Adjacent Sentences
        # Merge adjacent clusters if there is at least one clique containing  at least one segment from ci and one from ci+1."
        # If sentence[i] and sentence[i-1] share a common clique ID, they merge.
        
        grouped_indices: List[List[int]] = []
        current_indices: List[int] = [0]

        for i in range(1, len(sentences)):
            prev_idx = i - 1
            curr_idx = i
            
            # Intersection of clique memberships
            common_cliques = sentence_to_cliques[prev_idx].intersection(sentence_to_cliques[curr_idx])
            
            if common_cliques:
                current_indices.append(i)
            else:
                grouped_indices.append(current_indices)
                current_indices = [i]
        
        # Append the last group
        grouped_indices.append(current_indices)

        # Final Merging
        # Any remaining single-sentence clusters are merged with the nearest neighboring cluster
        final_indices = self._merge_singletons(grouped_indices, embeddings)

        # Construct Chunks
        chunks = []
        for i, group_idxs in enumerate(final_indices):
            chunk_text = " ".join([sentences[idx] for idx in group_idxs])
            new_chunk = Chunk(
                text=chunk_text,
                metadata={
                    **(document_meta or {}),
                    "chunk_index": i,
                    "method": "graph_chunker"
                },
            )
            chunks.append(new_chunk)

        return chunks

    def _merge_singletons(
        self, 
        groups_indices: List[List[int]], 
        embeddings: np.ndarray
    ) -> List[List[int]]:
        """
        Final Merging Phase:
        Any remaining single-sentence clusters are merged with the nearest neighboring 
        cluster based on cosine similarity.
        """
        if len(groups_indices) < 2:
            return groups_indices

        merged_groups = []
        i = 0
        
        while i < len(groups_indices):
            current_group = groups_indices[i]
            
            if len(current_group) > 1:
                merged_groups.append(current_group)
                i += 1
                continue

            # Finding an orphan sentence
            singleton_idx = current_group[0]
            singleton_vec = embeddings[singleton_idx].reshape(1, -1)

            # Calculate similarity to the LEFT neighbor (if exists)
            sim_left = -1.0
            if merged_groups:
                left_indices = merged_groups[-1]
                left_centroid = np.mean(embeddings[left_indices], axis=0).reshape(1, -1)
                sim_left = cosine_similarity(singleton_vec, left_centroid)[0][0]

            # Calculate similarity to the RIGHT neighbor (if exists)
            sim_right = -1.0
            if i + 1 < len(groups_indices):
                right_indices = groups_indices[i+1]
                right_centroid = np.mean(embeddings[right_indices], axis=0).reshape(1, -1)
                sim_right = cosine_similarity(singleton_vec, right_centroid)[0][0]

            # Decision based on higher similarity
            if sim_left == -1.0 and sim_right == -1.0:

                merged_groups.append(current_group)
                i += 1
            elif sim_left >= sim_right:
    
                merged_groups[-1].extend(current_group)
                i += 1
            else:

                combined_group = current_group + groups_indices[i+1]
                merged_groups.append(combined_group)
                # Skip the next group as it's now part of this one
                i += 2

        return merged_groups