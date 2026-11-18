"""
Unit and integration tests for Phase 5: Conflict-Aware Resolution (Module E) and Context-Budget Manager (Module F).
"""

from datetime import datetime, timezone
import pytest
from memo_exi.models.memory_models import (
    ExtractedFact,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    QueryIntent,
    RetrievalCandidate,
)
from memo_exi.modules.budget.budget_manager import ContextBudgetManager
from memo_exi.modules.conflict.resolver import ConflictResolver


@pytest.fixture
def conflict_resolver():
    return ConflictResolver()


@pytest.fixture
def budget_manager():
    return ContextBudgetManager(
        weight_relevance=0.45,
        weight_temporal=0.25,
        weight_importance=0.20,
        weight_redundancy=0.10,
        default_token_budget=1024,
    )


def test_conflict_resolver_legitimate_update(conflict_resolver):
    """Verify that updates with transition cues are recognized as valid updates, not conflicts."""
    existing = MemoryRecord(
        subject="User",
        attribute="city",
        value="Seattle",
        status=MemoryStatus.CURRENT,
    )
    incoming = ExtractedFact(
        subject="User",
        attribute="city",
        value="New York",
        raw_snippet="I moved to New York last week.",
    )

    res = conflict_resolver.analyze_update(existing, incoming)
    assert res.is_legitimate_update is True
    assert res.is_contradiction is False
    assert res.requires_clarification is False


def test_conflict_resolver_direct_contradiction(conflict_resolver):
    """Verify that opposing assertions without transition cues trigger contradiction workflow."""
    existing = MemoryRecord(
        subject="User",
        attribute="diet",
        value="Vegetarian",
        status=MemoryStatus.CURRENT,
    )
    incoming = ExtractedFact(
        subject="User",
        attribute="diet",
        value="Meat-eater",
        raw_snippet="I love eating steaks and meat daily.",
    )

    res = conflict_resolver.analyze_update(existing, incoming)
    assert res.is_contradiction is True
    assert res.requires_clarification is True
    assert res.clarification_question is not None
    assert "diet" in res.clarification_question.lower()


def test_budget_manager_utility_ranking(budget_manager):
    """Verify candidate ranking in descending order of utility."""
    rec1 = MemoryRecord(subject="User", attribute="role", value="Staff Engineer", importance_score=0.9)
    rec2 = MemoryRecord(subject="User", attribute="color", value="Blue", importance_score=0.2)

    c1 = RetrievalCandidate(memory=rec1, relevance_score=0.9, temporal_score=1.0, importance_score=0.9, estimated_tokens=10)
    c2 = RetrievalCandidate(memory=rec2, relevance_score=0.3, temporal_score=0.5, importance_score=0.2, estimated_tokens=10)

    result = budget_manager.optimize_selection(
        query="What is my job?",
        intent=QueryIntent.CURRENT_STATE,
        candidates=[c2, c1],
        token_budget=100,
    )

    # c1 has higher relevance and importance, should rank first
    assert len(result.selected_memories) == 2
    assert result.selected_memories[0].attribute == "role"
    assert result.candidates[0].utility_score > result.candidates[1].utility_score


def test_budget_manager_strict_token_budget_enforcement(budget_manager):
    """Verify that candidate selection strictly respects a tight token budget."""
    candidates = []
    for i in range(10):
        rec = MemoryRecord(subject="User", attribute=f"attr_{i}", value=f"val_{i}", importance_score=0.5)
        # Each item estimated at 15 tokens
        c = RetrievalCandidate(
            memory=rec,
            relevance_score=0.8 - (i * 0.05),
            temporal_score=1.0,
            importance_score=0.5,
            estimated_tokens=15,
        )
        candidates.append(c)

    # Budget of only 35 tokens -> can only select 2 items (15 + 15 = 30 tokens <= 35)
    result = budget_manager.optimize_selection(
        query="test query",
        intent=QueryIntent.CURRENT_STATE,
        candidates=candidates,
        token_budget=35,
    )

    assert result.budget_compliant is True
    assert result.total_tokens_used <= 35
    assert len(result.selected_memories) == 2
    assert result.total_tokens_used == 30
