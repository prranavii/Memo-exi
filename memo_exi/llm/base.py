"""
Abstract Base Class for LLM providers in MEMO-EXI.
Enables pluggable LLM backends (Groq, OpenAI, Anthropic, Mock).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMProvider(ABC):
    """Abstract interface for LLM client integration."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a completion for the given prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider credentials and network connection are active."""
        pass
