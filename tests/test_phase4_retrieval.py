"""
Unit and integration tests for Phase 4: Query Classification (Module C) and Semantic Retrieval (Module D).
"""

from datetime import datetime, timezone
import pytest
from memo_exi.llm.mock_provider import MockLLMProvider
from memo_exi.models.memory_models import ExtractedFact, MemoryStatus, MemoryType, QueryIntent
from memo_exi.modules.classification.classifier import TemporalQueryClassifier
from memo_exi.modules.retrieval.retriever import SemanticTemporalRetriever
from memo_exi.modules.temporal.manager import TemporalMemoryManager


@pytest.fixture
def classifier():
    return TemporalQueryClassifier(llm_provider=MockLLMProvider())


@pytest.fixture
def retriever(sqlite_manager, chroma_manager):
    return SemanticTemporalRetriever(sqlite_manager=sqlite_manager, chroma_manager=chroma_manager)


def test_query_classification_intents(classifier):
    """Verify classification across all 5 query intents."""
    ref_time = datetime(2024, 11, 1, tzinfo=timezone.utc)

    # 1. Current state
    res_curr = classifier.classify_query("What is my current salary?", reference_time=ref_time)
    assert res_curr.intent == QueryIntent.CURRENT_STATE

    # 2. Historical query with date constraint
    res_hist = classifier.classify_query("What was my salary before October 2024?", reference_time=ref_time)
    assert res_hist.intent == QueryIntent.HISTORICAL
    assert res_hist.target_datetime is not None
    assert res_hist.target_datetime.month == 10
    assert res_hist.target_datetime.year == 2024

    # 3. Temporal evolution / lineage query
    res_temp = classifier.classify_query("When did my project deadline change?", reference_time=ref_time)
    assert res_temp.intent == QueryIntent.TEMPORAL

    # 4. General semantic query
    res_gen = classifier.classify_query("Tell me about my favorite programming languages", reference_time=ref_time)
    assert res_gen.intent == QueryIntent.GENERAL_SEMANTIC

    # 5. Ambiguous query
    res_amb = classifier.classify_query("Why?", reference_time=ref_time)
    assert res_amb.intent == QueryIntent.AMBIGUOUS


def test_current_state_retrieval_filtering(retriever, sqlite_manager, chroma_manager):
    """Verify that current-state queries strictly exclude superseded versions."""
    temporal = TemporalMemoryManager(sqlite_manager=sqlite_manager, chroma_manager=chroma_manager)

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2024, 6, 1, tzinfo=timezone.utc)

    # Ingest old salary
    temporal.ingest_fact(
        ExtractedFact(subject="User", attribute="salary", value="$80,000", valid_from=t0),
        session_id="s1",
    )
    # Update to new salary
    temporal.ingest_fact(
        ExtractedFact(subject="User", attribute="salary", value="$110,000", valid_from=t1),
        session_id="s1",
    )

    classifier = TemporalQueryClassifier(llm_provider=MockLLMProvider())
    classification = classifier.classify_query("What is my current salary?")

    candidates = retriever.retrieve("What is my current salary?", classification, session_id="s1", top_k=5)
    assert len(candidates) > 0

    # Every candidate returned must be status='current'
    for c in candidates:
        assert c.memory.status == MemoryStatus.CURRENT
    # Value must be $110,000, not $80,000
    assert any(c.memory.value == "$110,000" for c in candidates)
    assert not any(c.memory.value == "$80,000" for c in candidates)


def test_historical_point_in_time_retrieval(retriever, sqlite_manager, chroma_manager):
    """Verify that historical queries correctly retrieve point-in-time state."""
    temporal = TemporalMemoryManager(sqlite_manager=sqlite_manager, chroma_manager=chroma_manager)

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2024, 6, 1, tzinfo=timezone.utc)

    temporal.ingest_fact(
        ExtractedFact(subject="User", attribute="role", value="Junior Analyst", valid_from=t0),
        session_id="s1",
    )
    temporal.ingest_fact(
        ExtractedFact(subject="User", attribute="role", value="Senior Analyst", valid_from=t1),
        session_id="s1",
    )

    classifier = TemporalQueryClassifier(llm_provider=MockLLMProvider())
    classification = classifier.classify_query("What was my role in March 2024?")

    candidates = retriever.retrieve("What was my role in March 2024?", classification, session_id="s1", top_k=5)
    assert len(candidates) > 0

    # Junior Analyst was valid in March 2024
    junior_cands = [c for c in candidates if c.memory.value == "Junior Analyst"]
    assert len(junior_cands) > 0
    assert junior_cands[0].temporal_score == 1.0


def test_temporal_lineage_retrieval(retriever, sqlite_manager, chroma_manager):
    """Verify that lineage chains are fetched for temporal evolution queries."""
    temporal = TemporalMemoryManager(sqlite_manager=sqlite_manager, chroma_manager=chroma_manager)

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2024, 5, 1, tzinfo=timezone.utc)

    temporal.ingest_fact(
        ExtractedFact(subject="Project", attribute="deadline", value="June 1", valid_from=t0),
        session_id="s1",
    )
    temporal.ingest_fact(
        ExtractedFact(subject="Project", attribute="deadline", value="August 15", valid_from=t1),
        session_id="s1",
    )

    classifier = TemporalQueryClassifier(llm_provider=MockLLMProvider())
    classification = classifier.classify_query("When did my project deadline change?")

    candidates = retriever.retrieve("When did my project deadline change?", classification, session_id="s1", top_k=5)
    chains = retriever.retrieve_lineage_chains(candidates)

    assert "Project::deadline" in chains
    history = chains["Project::deadline"]
    assert len(history) == 2
    assert [h.value for h in history] == ["June 1", "August 15"]
