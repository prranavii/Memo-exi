"""
Module G: Adaptive Memory Controller.
Dynamic decision-making policy engine that adapts retrieval parameters,
query reformulation, filtering strategies, and memory selection.
Designed modularly to allow future learned reinforcement policies.
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from memo_exi.config import settings
from memo_exi.models.memory_models import QueryIntent, RetrievalCandidate

logger = logging.getLogger(__name__)


class RetrievalPolicy(BaseModel):
    """Dynamic policy parameters chosen by the controller."""
    top_k: int
    filter_strategy: str  # "strict_current", "temporal_window", "lineage_trace", "all_open"
    allow_reformulation: bool = True
    context_budget: int = 1024
    min_similarity_threshold: float = 0.25
    decision_rationale: str


class ControllerDecision(BaseModel):
    """Log record of adaptive actions taken during query execution."""
    initial_policy: RetrievalPolicy
    reformulated_query: Optional[str] = None
    retry_executed: bool = False
    final_candidate_count: int = 0
    actions_taken: List[str] = Field(default_factory=list)


class AdaptiveMemoryController:
    """Modular controller deciding memory operations dynamically."""

    def __init__(
        self,
        default_budget: Optional[int] = None,
        default_top_k: Optional[int] = None,
    ):
        self.default_budget = default_budget or settings.default_token_budget
        self.default_top_k = default_top_k or settings.default_top_k

    def formulate_policy(
        self,
        intent: QueryIntent,
        total_memories: int,
        context_budget: Optional[int] = None,
    ) -> RetrievalPolicy:
        """
        Dynamically determine retrieval policy based on intent, memory density, and budget.
        """
        budget = context_budget or self.default_budget
        actions = []

        # 1. Density-aware top_k adjustment:
        # If memory store is sparse (< 5 memories), search smaller k to minimize noise
        if total_memories <= 3:
            k = min(total_memories, 3)
            density_note = "Sparse memory density: limited top-k."
        elif total_memories < 20:
            k = min(self.default_top_k, total_memories)
            density_note = "Moderate memory density: standard top-k."
        else:
            k = min(settings.max_top_k, self.default_top_k + 3)
            density_note = "High memory density: expanded candidate retrieval."

        # 2. Budget pressure adjustment:
        # If budget is tight (< 512 tokens), restrict k to prevent budget waste
        if budget < 512:
            k = min(k, 3)
            budget_note = "Context pressure active: constrained top-k."
        else:
            budget_note = "Context budget adequate."

        # 3. Intent-based routing:
        if intent == QueryIntent.CURRENT_STATE:
            strategy = "strict_current"
            rationale = f"Current-state query intent. Enforcing strict active status filter. {density_note} {budget_note}"
        elif intent == QueryIntent.HISTORICAL:
            strategy = "temporal_window"
            k = max(k, 5)  # Historical queries need depth across versions
            rationale = f"Historical query intent. Enabling point-in-time window evaluation. {density_note} {budget_note}"
        elif intent == QueryIntent.TEMPORAL:
            strategy = "lineage_trace"
            k = max(k, 6)
            rationale = f"Temporal evolution query intent. Enabling full lineage tracking. {density_note} {budget_note}"
        elif intent == QueryIntent.AMBIGUOUS:
            strategy = "all_open"
            k = max(k, 5)
            rationale = f"Ambiguous query intent. Retrieving diverse cross-topic candidates. {density_note} {budget_note}"
        else:
            strategy = "all_open"
            rationale = f"General semantic query intent. Ranking across all candidate memories. {density_note} {budget_note}"

        return RetrievalPolicy(
            top_k=max(k, 1),
            filter_strategy=strategy,
            allow_reformulation=True,
            context_budget=budget,
            decision_rationale=rationale,
        )

    def evaluate_retrieval_and_adapt(
        self,
        query: str,
        intent: QueryIntent,
        candidates: List[RetrievalCandidate],
    ) -> Optional[str]:
        """
        Inspect retrieval quality. If zero candidates or all similarity scores are poor,
        formulate a broader query reformulation for retry.
        """
        # If we found high quality results, no need to reformulate
        if candidates and any(c.relevance_score >= 0.35 for c in candidates):
            return None

        # Query Reformulation heuristic
        # Strip question phrasing to keywords: e.g. "What was my salary before October?" -> "salary"
        stopwords = {
            "what", "is", "my", "was", "the", "were", "where", "did", "i", "do",
            "when", "how", "why", "who", "which", "are", "tell", "me", "about",
            "current", "currently", "now", "before", "in", "at", "to", "for",
        }
        words = [w.strip("?,.!") for w in query.lower().split() if w.strip("?,.!")]
        content_words = [w for w in words if w not in stopwords and len(w) > 2]

        if content_words:
            reformulated = " ".join(content_words)
            logger.info(f"Adaptive Controller triggered query reformulation: '{query}' -> '{reformulated}'")
            return reformulated

        return None
