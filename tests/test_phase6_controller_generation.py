"""
Unit and integration tests for Phase 6: Adaptive Controller (Module G) and LLM Response Generation (Module H).
"""

from datetime import datetime, timezone
import pytest
from memo_exi.llm.mock_provider import MockLLMProvider
from memo_exi.models.memory_models import (
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    QueryIntent,
    RetrievalCandidate,
    RetrievalResult,
)
from memo_exi.modules.controller.controller import AdaptiveMemoryController
from memo_exi.modules.generation.generator import ResponseGenerator


@pytest.fixture
def controller():
    return AdaptiveMemoryController(default_budget=1024, default_top_k=5)


@pytest.fixture
def generator():
    return ResponseGenerator(llm_provider=MockLLMProvider())


def test_controller_policy_sparse_vs_dense(controller):
    """Verify that controller dynamically adapts top-k to memory density."""
    # Sparse memory (2 memories total)
    policy_sparse = controller.formulate_policy(
        intent=QueryIntent.CURRENT_STATE,
        total_memories=2,
    )
    assert policy_sparse.top_k <= 2
    assert policy_sparse.filter_strategy == "strict_current"

    # Dense memory (50 memories)
    policy_dense = controller.formulate_policy(
        intent=QueryIntent.CURRENT_STATE,
        total_memories=50,
    )
    assert policy_dense.top_k > 5


def test_controller_context_pressure(controller):
    """Verify that tight context budgets constrain retrieval depth."""
    policy_tight = controller.formulate_policy(
        intent=QueryIntent.GENERAL_SEMANTIC,
        total_memories=50,
        context_budget=256,
    )
    assert policy_tight.top_k <= 3


def test_controller_query_reformulation(controller):
    """Verify that poor or failed retrievals trigger query reformulation."""
    bad_candidates = [
        RetrievalCandidate(
            memory=MemoryRecord(subject="User", attribute="other", value="val"),
            relevance_score=0.15,  # below quality threshold
        )
    ]
    reformulated = controller.evaluate_retrieval_and_adapt(
        query="What was my salary before October?",
        intent=QueryIntent.HISTORICAL,
        candidates=bad_candidates,
    )
    assert reformulated is not None
    assert "salary" in reformulated.lower()


def test_generator_current_state(generator):
    """Verify generated response cites current active memory."""
    rec_current = MemoryRecord(
        subject="User",
        attribute="salary",
        value="$120,000",
        status=MemoryStatus.CURRENT,
    )
    retrieval_res = RetrievalResult(
        query="What is my current salary?",
        intent=QueryIntent.CURRENT_STATE,
        selected_memories=[rec_current],
        token_budget=1024,
    )

    resp = generator.generate_response("What is my current salary?", retrieval_res)
    assert resp.grounded is True
    assert len(resp.memories_cited) == 1
    assert not resp.has_disputed_facts


def test_generator_missing_information(generator):
    """Verify response for missing information clearly communicates lack of records."""
    retrieval_res = RetrievalResult(
        query="What is my passport number?",
        intent=QueryIntent.CURRENT_STATE,
        selected_memories=[],
        token_budget=1024,
    )
    resp = generator.generate_response("What is my passport number?", retrieval_res)
    assert "no" in resp.answer.lower() or "not have" in resp.answer.lower()
    assert len(resp.memories_cited) == 0


def test_generator_disputed_fact_warning(generator):
    """Verify that disputed facts are flagged in the response metadata."""
    rec_disputed = MemoryRecord(
        subject="User",
        attribute="city",
        value="Chicago",
        status=MemoryStatus.DISPUTED,
    )
    retrieval_res = RetrievalResult(
        query="Where do I live?",
        intent=QueryIntent.CURRENT_STATE,
        selected_memories=[rec_disputed],
        token_budget=1024,
    )
    resp = generator.generate_response("Where do I live?", retrieval_res)
    assert resp.has_disputed_facts is True
