"""
LLM Provider Factory.
Returns configured LLM provider (Groq or Mock fallback).
"""

import logging
from memo_exi.config import settings
from memo_exi.llm.base import LLMProvider
from memo_exi.llm.groq_provider import GroqProvider
from memo_exi.llm.mock_provider import MockLLMProvider

logger = logging.getLogger(__name__)


def get_llm_provider(force_mock: bool = False) -> LLMProvider:
    """
    Get the active LLM provider.
    If Groq is configured and available, returns GroqProvider;
    otherwise returns MockLLMProvider for offline reliability and testing.
    """
    if force_mock:
        return MockLLMProvider()

    groq = GroqProvider()
    if groq.is_available():
        return groq

    logger.info("Using MockLLMProvider (Groq not configured or unavailable)")
    return MockLLMProvider()
