from typing import List, Dict, Any, Optional
from ..core.base import BaseChunker, Chunk
from ..core.progress import coerce_progress_enabled, iter_with_progress
from ..core.registry import chunker
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import textwrap
from nltk.tokenize import sent_tokenize


@chunker("lumberchunker")
class LumberChunker(BaseChunker):
    '''
    LumberChunker method was adopted from the original implementation: "https://github.com/joaodsmarques/LumberChunker".
    Instead of gpt-3.5-turbo-0125 and gemini-pro it uses locally hosted Qwen3-4B-Instruct-2507 model from HuggingFace.
    Qwen was chosen as the default model following another LumberChunker adaptation: "https://github.com/IAAR-Shanghai/Meta-Chunking/blob/main/MoC/LumberChunker.py".
    Params:
        model: HuggingFace model ID to be used for chunking (default: Qwen/Qwen3-4B-Instruct-2507)
        group_size_threshold: Maximum number of words per chunk (default: 350)
        max_retries: Maximum number of retries for LLM calls (default: 3)
        sleep_seconds: Sleep time between retries for LLM calls (default: 20)
        max_new_tokens: Maximum number of new tokens to generate in LLM calls (default 100)
    '''

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self._system_prompt = textwrap.dedent("""\
            You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

            Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

            Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX', without any explanatory notes.

            Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""
        )

        config = config or {}

        model_id = config["model"] if "model" in config else "Qwen/Qwen3-4B-Instruct-2507"
        self._group_size_threshold = int(config["group_size_threshold"]) if "group_size_threshold" in config else 350
        self._max_retries = int(config["max_retries"]) if "max_retries" in config else 3
        self._sleep_seconds = int(config["sleep_seconds"]) if "sleep_seconds" in config else 20
        self._max_new_tokens = int(config["max_new_tokens"]) if "max_new_tokens" in config else 100
        
        if self._group_size_threshold <= 0:
            raise ValueError("group_size_threshold must be positive")
        if self._max_retries <= 0:
            raise ValueError("max_retries must be positive")
        if self._sleep_seconds < 0:
            raise ValueError("sleep_seconds must be non-negative")
        if self._max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")     
        
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
        
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            token=token
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
            iter_with_progress(documents, desc="LumberChunker Chunking", enabled=show_progress)
        ):
            meta = documents_meta[idx] if documents_meta is not None else None
            all_chunks.extend(self._split_single(text, meta))

        return all_chunks

    def _split_single(
        self,
        text: str,
        document_meta: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        chunks = []

        # Add ID to each sentence
        full_segments = sent_tokenize(text)
        full_segments = [f"ID {idx}: {seg}" for idx, seg in enumerate(full_segments)]
        
        chunk_number = 0
        i = 0
        new_id_list = []

        # Mostly following the original implementation logic here
        while chunk_number < len(full_segments)-5:
            word_count = 0
            i = 0
            while word_count < self._group_size_threshold  and i+chunk_number<len(full_segments)-1:
                i += 1
                final_document = "\n".join(f"{full_segments[k]}" for k in range(chunk_number, i + chunk_number))
                word_count = self._count_words(final_document)
            
            if(i == 1):
                final_document = "\n".join(f"{full_segments[k]}" for k in range(chunk_number, i + chunk_number))
            else:
                final_document = "\n".join(f"{full_segments[k]}" for k in range(chunk_number, i-1 + chunk_number))
            
            
            question = f"\nDocument:\n{final_document}"

            chunk_number = chunk_number + i-1

            try:
                llm_output = self._LLM_prompt(user_prompt=question, max_retries=self._max_retries, sleep_seconds=self._sleep_seconds)
            except RuntimeError as e:
                print(f"LLM prompt failed for chunk starting at ID {chunk_number}. Error: {e}")
                chunk_number = chunk_number + 1
                continue

            # For books where there is dubious content, Gemini refuses to run the prompt and returns mistake. This is to avoid being stalled here forever.
            if llm_output == "content_flag_increment":
                chunk_number = chunk_number + 1
            else:
                pattern = r"Answer: ID \w+"
                match = re.search(pattern, llm_output)

                if match == None:
                    # print("repeat this one")
                    # Differently from original implementation in case of no match, we just increment by 1 to avoid being stuck
                    chunk_number = chunk_number + 1
                else:
                    gpt_output1 = match.group(0)
                    pattern = r'\d+'
                    match = re.search(pattern, gpt_output1)
                    chunk_number = int(match.group())
                    new_id_list.append(chunk_number)
                    chunk_number = chunk_number + 1

        #Add the last chunk to the list
        new_id_list.append(len(full_segments))

        # Remove IDs as they no longer make sense here.
        full_segments = [re.sub(r'^ID \d+:\s*', '', doc) for doc in full_segments]

        for i in range(len(new_id_list)):
            # Calculate the start and end indices of each chunk
            start_idx = new_id_list[i-1] if i > 0 else 0
            end_idx = new_id_list[i]

            chunk_text = "\n".join(full_segments[start_idx:end_idx])
            new_chunk = Chunk(
                text = chunk_text,
                metadata={
                    **document_meta
                },
            )
            chunks.append(new_chunk)

        return chunks

    # Count_Words idea is to approximate the number of tokens in the sentence. We are assuming 1 word ~ 1.2 Tokens
    def _count_words(self, input_string):
        words = input_string.split()
        return round(1.2*len(words))

    def _LLM_prompt(self, user_prompt, max_retries, sleep_seconds):
        last_exception = None

        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize = False,
                    add_generation_prompt = True,
                    enable_thinking = False
                )

                model_inputs = self._tokenizer(
                    [text],
                    return_tensors="pt"
                ).to(self._model.device)

                generated_ids = self._model.generate(
                    **model_inputs,
                    max_new_tokens = self._max_new_tokens
                )

                generated_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = self._tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens = True
                )[0].strip()

                return response

            except Exception as e:
                last_exception = e
                print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(sleep_seconds)

        raise RuntimeError("LLM prompt failed after retries") from last_exception
