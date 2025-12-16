from typing import Callable, Dict

from src.evaluation.retrieval import TfidfRetriever

RetrieverFactory = Callable[[], object]

_RETRIEVER_REGISTRY: Dict[str, RetrieverFactory] = {}


def register_retriever(name: str, factory: RetrieverFactory):
    if name in _RETRIEVER_REGISTRY:
        raise ValueError(f"Retriever '{name}' already registered")
    _RETRIEVER_REGISTRY[name] = factory


def get_retriever(name: str):
    factory = _RETRIEVER_REGISTRY.get(name)
    if not factory:
        raise ValueError(f"Unknown retriever '{name}'. Registered: {list(_RETRIEVER_REGISTRY.keys())}")
    return factory()


def list_retrievers():
    return list(_RETRIEVER_REGISTRY.keys())


# register built-ins
register_retriever("tfidf", lambda: TfidfRetriever())
