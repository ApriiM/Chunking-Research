import json
import random
import copy
from typing import List, Dict, Tuple, Any

class DocumentMerger:
    def __init__(self, seed: int = 14, separator: str = "\n\n"):
        """
        Initializes the merger with a fixed seed.
        """
        self.seed = seed
        self.separator = separator
        random.seed(self.seed)

    def merge_dataset(self, documents: List[Dict], queries: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Merges documents and updates query metadata.
        Uses original_ prefix for legacy data and document_start/end for new offsets.
        """
        
        target_doc_id = "merged_doc_001"
        
        # Shuffle documents
        shuffled_docs = documents.copy()
        random.shuffle(shuffled_docs)
        
        merged_text = ""
        doc_offsets = {}  # Map: original_doc_id -> {start, end}
        
        print(f"[Merger] Processing {len(shuffled_docs)} documents into '{target_doc_id}'...")

        # Merge Documents Loop
        for doc in shuffled_docs:
            doc_id = doc.get("id") or doc.get("doc_id")
            content = doc.get("contents") or doc.get("text", "")
            
            start_index = len(merged_text)
            merged_text += content
            end_index = len(merged_text)
            
            # Store location in the new mega-file
            doc_offsets[doc_id] = {
                "start": start_index,
                "end": end_index
            }
            
            merged_text += self.separator

        # Create the new Mega-Document
        merged_documents_output = [{
            "id": target_doc_id,
            "contents": merged_text,
            "metadata": {
                "source": "synthetic_merge",
                "seed": self.seed,
                "original_doc_count": len(documents)
            }
        }]

        # Remap Queries Loop
        remapped_queries = []
        
        for q in queries:
            if not q.get("relevant"):
                continue
            
            original_doc_id = q["relevant"][0]
            
            if original_doc_id not in doc_offsets:
                continue
                
            offset_info = doc_offsets[original_doc_id]
            
            new_q = copy.deepcopy(q)
            old_meta = new_q.get("metadata", {})
            new_q["metadata"]["original_document_id"] = original_doc_id
            new_q["relevant"] = [target_doc_id]
            
            # Add document boundaries in the new file 
            new_q["metadata"]["document_start"] = offset_info["start"]
            new_q["metadata"]["document_end"] = offset_info["end"]
            
            # Rename original answer fields to be explicit
            if "answer_starts" in old_meta:
                new_q["metadata"]["original_answer_starts"] = old_meta.pop("answer_starts")
            
            if "answers" in old_meta:
                new_q["metadata"]["original_answers"] = old_meta.pop("answers")

            remapped_queries.append(new_q)

        return merged_documents_output, remapped_queries