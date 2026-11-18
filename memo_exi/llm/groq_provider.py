"""
Groq API integration provider for MEMO-EXI.
Supports high-throughput JSON-mode inference with Llama 3 models.
"""

import logging
from typing import Optional
from memo_exi.config import settings
from memo_exi.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Groq API client implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if self.api_key and self.api_key not in ("mock_key_or_placeholder", "your_groq_api_key_here"):
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
                logger.info(f"Initialized Groq client with model: {self.model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
                self._client = None
        else:
            logger.info("Groq API key not configured or set to placeholder.")

    def is_available(self) -> bool:
        return self._client is not None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self._client:
            raise RuntimeError(
                "GroqProvider is not available. Please configure a valid GROQ_API_KEY in .env."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else settings.llm_temperature
        tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq generation failed: {e}", exc_info=True)
            raise
