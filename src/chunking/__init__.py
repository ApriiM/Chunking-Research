# add more chunkers here as needed
from .strategies.fixed_size import FixedSizeChunker
from .strategies.passage import SentencePassageChunker
from .strategies.text_tiling import TextTilingChunker
from .defaults import merge_with_defaults


def get_chunker(chunker_name: str, config: dict):
    """
    Factory function to initialize a chunker based on its name.
    """
    chunkers = {
        # add more chunkers here as needed
        "fixed_size": FixedSizeChunker,
        "passage": SentencePassageChunker,
        "text_tiling": TextTilingChunker,
    }
    
    if chunker_name not in chunkers:
        raise ValueError(f"Chunker '{chunker_name}' not found. Available: {list(chunkers.keys())}")

    merged_config = merge_with_defaults(chunker_name, config or {})
    return chunkers[chunker_name](merged_config)