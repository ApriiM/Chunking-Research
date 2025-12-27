from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid

@dataclass
class Chunk:
    '''
    Represents a text chunk with associated metadata.
    ''' 
    text: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseChunker(ABC):
    '''
    Abstract base class for text chunking strategies.
    '''
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def split_text(
        self,
        documents: List[str],
        documents_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Chunk]:
        '''
        Split a collection of documents into an ordered list of `Chunk` objects.

        :param documents: Raw documents to segment
        :type documents: List[str]
        :param documents_meta: Optional list of metadata, aligned to `documents`
        :type documents_meta: Optional[List[Dict[str, Any]]]
        :return: Ordered list of chunks produced by the strategy for all documents
        :rtype: List[Chunk]
        '''
        pass