from typing import List, Tuple

from src.data_loader.core.schemas import DocumentRecord, QueryRecord

# Dataset loaders should return a pair of document and query records already
# normalized to the unified JSONL shapes.
DatasetArtifacts = Tuple[List[DocumentRecord], List[QueryRecord]]
