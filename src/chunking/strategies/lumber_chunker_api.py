import os
import time
import textwrap
from typing import Any, Dict, Optional

from ..core.base import BaseChunker
from ..core.progress import coerce_progress_enabled
from ..core.registry import chunker
from .lumberchunker import LumberChunker


def _load_env_file(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


@chunker("lumberchunker_api")
@chunker("lumber_chunker_api")
class LumberChunkerAPI(LumberChunker):
    """
    API-based LumberChunker variant.

    Reuses the existing LumberChunker segmentation loop (adapted from the
    original submodule), but swaps local HF generation for API calls.

    Supported providers:
    - openai
    - gemini
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        BaseChunker.__init__(self, config)
        cfg = config or {}

        # Optional .env loading (before resolving env-backed settings).
        dotenv_path = str(cfg.get("dotenv_path", ".env"))
        load_dotenv = coerce_progress_enabled(cfg.get("load_dotenv"), default=True)
        if load_dotenv:
            _load_env_file(dotenv_path)

        # Keep the same prompt used in local LumberChunker.
        self._system_prompt = textwrap.dedent(
            """\
            You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

            Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

            Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX', without any explanatory notes.

            Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""
        )

        # Reuse attribute names expected by LumberChunker._split_single.
        self._group_size_threshold = int(cfg.get("group_size_threshold", 350))
        self._max_retries = int(cfg.get("max_retries", 3))
        self._sleep_seconds = int(cfg.get("sleep_seconds", 20))
        self._max_new_tokens = int(cfg.get("max_output_tokens", cfg.get("max_new_tokens", 100)))

        if self._group_size_threshold <= 0:
            raise ValueError("group_size_threshold must be positive")
        if self._max_retries <= 0:
            raise ValueError("max_retries must be positive")
        if self._sleep_seconds < 0:
            raise ValueError("sleep_seconds must be non-negative")
        if self._max_new_tokens <= 0:
            raise ValueError("max_output_tokens/max_new_tokens must be positive")

        self.provider = str(
            cfg.get("provider", os.environ.get("LUMBERCHUNKER_API_PROVIDER", "openai"))
        ).strip().lower()
        if self.provider not in {"openai", "gemini"}:
            raise ValueError("provider must be one of: openai, gemini")

        # Provider-specific request settings.
        self.model = str(
            cfg.get("model", os.environ.get("LUMBERCHUNKER_API_MODEL", "gpt-4o-mini"))
        )
        self.temperature = float(cfg.get("temperature", 0.1))
        self.request_timeout_seconds = int(cfg.get("request_timeout_seconds", 120))
        self.base_url = cfg.get("base_url", os.environ.get("LUMBERCHUNKER_API_BASE_URL"))
        self.api_key = cfg.get("api_key")
        self.api_key_env_var = cfg.get("api_key_env_var")

        self.api_key = self.api_key or self._resolve_api_key_from_env()
        if not self.api_key:
            raise ValueError(
                f"Missing API key for provider '{self.provider}'. "
                "Set config.api_key or relevant env var."
            )

        self._openai_client = None
        self._gemini_model = None

    def _resolve_api_key_from_env(self) -> Optional[str]:
        if self.api_key_env_var:
            return os.environ.get(str(self.api_key_env_var))
        if self.provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    # Signature matches LumberChunker._split_single expectations.
    def _LLM_prompt(self, user_prompt, max_retries, sleep_seconds):
        last_exception = None

        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    return self._openai_prompt(user_prompt)
                return self._gemini_prompt(user_prompt)
            except Exception as e:
                last_exception = e
                print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(sleep_seconds)

        raise RuntimeError("LLM prompt failed after retries") from last_exception

    def _openai_prompt(self, user_prompt: str) -> str:
        if self._openai_client is None:
            try:
                from openai import OpenAI
            except Exception as exc:
                raise ImportError(
                    "OpenAI provider selected but package 'openai' is not installed. "
                    "Install with: pip install openai"
                ) from exc
            self._openai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        completion = self._openai_client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self._max_new_tokens,
            timeout=self.request_timeout_seconds,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = completion.choices[0].message.content
        return content if content is not None else ""

    def _gemini_prompt(self, user_prompt: str) -> str:
        if self._gemini_model is None:
            try:
                import google.generativeai as genai
            except Exception as exc:
                raise ImportError(
                    "Gemini provider selected but package 'google-generativeai' is not installed. "
                    "Install with: pip install google-generativeai"
                ) from exc
            genai.configure(api_key=self.api_key)
            self._gemini_model = genai.GenerativeModel(self.model)

        response = self._gemini_model.generate_content(
            f"{self._system_prompt}\n\n{user_prompt}",
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self._max_new_tokens,
            },
            request_options={"timeout": self.request_timeout_seconds},
        )

        text = getattr(response, "text", None)
        if text:
            return text
        # Keep compatibility with original "blocked/empty" handling path.
        return "content_flag_increment"
