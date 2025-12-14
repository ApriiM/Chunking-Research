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
    def split_text(self, text: str, document_meta: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        '''
        Split input text into a list of `Chunk` objects.

        :param text: Raw text to segment
        :type text: str
        :param document_meta: Optional metadata propagated into each chunk
        :type document_meta: Optional[Dict[str, Any]]
        :return: Ordered list of chunks produced by the strategy
        :rtype: List[Chunk]
        '''
        pass