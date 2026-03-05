from __future__ import annotations

import math
from string import Template
from typing import List, Dict, Any, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..core.base import BaseChunker, Chunk
from ..core.registry import chunker


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "Qwen/Qwen3-7B"
_DEFAULT_CONTEXT_TOKENS = 32_768
_OVERLAP_RATIO = 0.05
_PROMPT_OVERHEAD_TOKENS = 700
_MAX_NEW_TOKENS = 2_048


# ---------------------------------------------------------------------------
# Structured output schema (FORCED by Outlines)
# ---------------------------------------------------------------------------

class PropositionList(BaseModel):
    propositions: List[str] = Field(
        default_factory=list,
        description="Self-contained factual propositions extracted from the input text.",
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
"""Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
4. Present the results as a list of strings, formatted in JSON.

Example:

Input: Title: ¯Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content:
The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in
1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in
other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were
frequently seen in gardens in spring, and thus may have served as a convenient explanation for the
origin of the colored eggs hidden there for children. Alternatively, there is a European tradition
that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and
both occur on grassland and are first seen in the spring. In the nineteenth century the influence
of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.
German immigrants then exported the custom to Britain and America where it evolved into the
Easter Bunny."
Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in
1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of
medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until
the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about
the possible explanation for the connection between hares and the tradition during Easter", "Hares
were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation
for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition
that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both
hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth
century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular
throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to
Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in
Britain and America."]"""
)

_PROMPT_TEMPLATE = Template(
"""<|system|>
$system_prompt
<|user|>
Decompose the following text:

$text
<|assistant|>
"""
)


def _build_prompt(text: str) -> str:
    return _PROMPT_TEMPLATE.substitute(
        system_prompt=_SYSTEM_PROMPT.strip(),
        text=text.strip(),
    )


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

@chunker("dense_x_retrieval")
class DenseXRetrievalChunker(BaseChunker):
    """
    Dense X Retrieval chunker using structured decoding via Outlines.

    Guarantees:
        - Output ALWAYS matches PropositionList schema.
        - No JSON parsing.
        - No retry loop.
        - Deterministic structure.
    """
    # TODO: refactor clarin api, check and force stuctued output, maybe better prompt format, check .yaml config params
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.model_id: str = self.config.get("model", _DEFAULT_MODEL)
        self.max_context: int = int(
            self.config.get("max_context", _DEFAULT_CONTEXT_TOKENS)
        )
        self.max_new_tokens: int = int(
            self.config.get("max_new_tokens", _MAX_NEW_TOKENS)
        )
        self.batch_size: int = int(self.config.get("batch_size", 1))
        self._overlap_ratio: float = float(
            self.config.get("overlap_ratio", _OVERLAP_RATIO)
        )

        # Device
        device_config = self.config.get("device", "auto")

        if device_config == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_config

        # Dtype
        dtype_config = self.config.get("torch_dtype", "auto")
        if dtype_config == "float16":
            self.torch_dtype = torch.float16
        elif dtype_config == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif dtype_config == "float32":
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = None  # let HF decide

        # Load tokenizer + model
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
        )

        # Budget
        self._text_token_budget = (
            self.max_context - _PROMPT_OVERHEAD_TOKENS - self.max_new_tokens
        )

        if self._text_token_budget <= 0:
            raise ValueError(
                "Context window too small for prompt + generation budget."
            )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def split_text(
        self,
        documents: List[str],
        documents_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Chunk]:

        if documents_meta and len(documents_meta) != len(documents):
            raise ValueError("documents_meta length must match documents")

        all_chunks: List[Chunk] = []

        for idx, text in enumerate(documents):
            meta = documents_meta[idx] if documents_meta else {}
            all_chunks.extend(self._process_document(text, meta, idx))

        return all_chunks

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _sliding_window_split(
        self, text: str
    ) -> List[Tuple[str, int, int]]:

        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        total = len(token_ids)

        window = self._text_token_budget
        overlap = math.ceil(window * self._overlap_ratio)
        step = window - overlap

        segments: List[Tuple[str, int, int]] = []

        start_tok = 0
        while start_tok < total:
            end_tok = min(start_tok + window, total)

            segment_text = self._tokenizer.decode(
                token_ids[start_tok:end_tok],
                skip_special_tokens=True,
            )

            segments.append((segment_text, 0, 0))  # char offsets optional

            if end_tok == total:
                break

            start_tok += step

        return segments

    def _generate_segment(
        self,
        segment_text: str,
    ) -> PropositionList:

        prompt = _build_prompt(segment_text)

        result: PropositionList = _(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
        )

        return result

    def _process_document(
        self,
        text: str,
        meta: Dict[str, Any],
        doc_idx: int,
    ) -> List[Chunk]:

        token_count = self._count_tokens(text)

        if token_count <= self._text_token_budget:
            segments = [(text, 0, 0)]
        else:
            segments = self._sliding_window_split(text)

        chunks: List[Chunk] = []
        global_prop_idx = 0

        for seg_idx, (segment, _, _) in enumerate(segments):

            result = self._generate_segment(segment)

            for proposition in result.propositions:
                if not proposition or not proposition.strip():
                    continue

                chunks.append(
                    Chunk(
                        text=proposition.strip(),
                        metadata={
                            **meta,
                            "source_document_index": doc_idx,
                            "proposition_index": global_prop_idx,
                            "window_segment_index": seg_idx,
                            "total_window_segments": len(segments),
                        },
                    )
                )

                global_prop_idx += 1

        return chunks