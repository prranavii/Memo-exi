"""
Unit and integration tests for Phase 2: Conversation Ingestion and Fact Extraction (Module A).
"""

from datetime import datetime, timezone
import pytest
from memo_exi.llm.mock_provider import MockLLMProvider
from memo_exi.models.memory_models import MemoryType
from memo_exi.modules.ingestion.extractor import FactExtractor


@pytest.fixture
def extractor():
    """Fact extractor powered by deterministic mock provider."""
    return FactExtractor(llm_provider=MockLLMProvider())


def test_extract_factual_salary(extractor):
    """Verify factual extraction of compensation/salary."""
    text = "My current salary is $115,000 per year."
    res = extractor.extract_facts(text, session_id="s1")

    assert not res.filtered_out_chitchat
    assert len(res.facts) >= 1
    fact = res.facts[0]
    assert fact.subject == "User"
    assert fact.attribute == "salary"
    assert "$115,000" in fact.value
    assert fact.memory_type == MemoryType.FACTUAL
    assert fact.confidence_score >= 0.8
    assert fact.importance_score >= 0.7


def test_extract_preference(extractor):
    """Verify preference memory type extraction."""
    text = "I prefer dark mode in VS Code."
    res = extractor.extract_facts(text, session_id="s1")

    assert not res.filtered_out_chitchat
    assert len(res.facts) >= 1
    fact = res.facts[0]
    assert fact.subject == "User"
    assert fact.memory_type == MemoryType.PREFERENCE
    assert "dark mode" in fact.value.lower()


def test_extract_constraint(extractor):
    """Verify constraint extraction."""
    text = "Always use 4 spaces for indentation, never use tabs."
    res = extractor.extract_facts(text, session_id="s1")

    assert not res.filtered_out_chitchat
    assert len(res.facts) >= 1
    fact = res.facts[0]
    assert fact.memory_type == MemoryType.CONSTRAINT


def test_filter_chitchat(extractor):
    """Verify that greetings and polite filler are filtered without generating memories."""
    greetings = [
        "Hello!",
        "Hi there, how are you?",
        "Good morning!",
        "Thanks a lot!",
    ]
    for msg in greetings:
        res = extractor.extract_facts(msg, session_id="s1")
        assert res.filtered_out_chitchat is True
        assert len(res.facts) == 0


def test_empty_input(extractor):
    """Verify handling of blank or whitespace input."""
    res = extractor.extract_facts("   ", session_id="s1")
    assert res.filtered_out_chitchat is True
    assert len(res.facts) == 0


def test_pydantic_schema_validity(extractor):
    """Verify that all extracted records strictly adhere to Pydantic constraints."""
    text = "The deadline for Project Phoenix was changed to 2024-12-15."
    res = extractor.extract_facts(text, session_id="s2")

    assert len(res.facts) == 1
    f = res.facts[0]
    # Check bounds
    assert 0.0 <= f.confidence_score <= 1.0
    assert 0.0 <= f.importance_score <= 1.0
    assert f.valid_from is not None
    assert f.memory_type in list(MemoryType)
