"""
Module F: Context-Budget Manager.
Ranks candidate memories using a parameterized utility function and selects
the optimal memory subset under a strict token budget.
"""

import logging
from typing import Dict, List, Optional, Set
from memo_exi.config import settings
from memo_exi.models.memory_models import (
    MemoryRecord,
    QueryIntent,
    RetrievalCandidate,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


class ContextBudgetManager:
    """Manages memory selection, utility ranking, and context-token budget enforcement."""

    def __init__(
        self,
        weight_relevance: Optional[float] = None,
        weight_temporal: Optional[float] = None,
        weight_importance: Optional[float] = None,
        weight_redundancy: Optional[float] = None,
        default_token_budget: Optional[int] = None,
    ):
        self.w_rel = weight_relevance if weight_relevance is not None else settings.weight_relevance
        self.w_temp = weight_temporal if weight_temporal is not None else settings.weight_temporal
        self.w_imp = weight_importance if weight_importance is not None else settings.weight_importance
        self.w_red = weight_redundancy if weight_redundancy is not None else settings.weight_redundancy
        self.default_budget = default_token_budget or settings.default_token_budget

    def optimize_selection(
        self,
        query: str,
        intent: QueryIntent,
        candidates: List[RetrievalCandidate],
        token_budget: Optional[int] = None,
        lineage_chains: Optional[Dict[str, List[MemoryRecord]]] = None,
    ) -> RetrievalResult:
        """
        Rank candidate memories by utility and select items greedily until token budget saturation.
        """
        budget = token_budget if token_budget is not None else self.default_budget
        if not candidates:
            return RetrievalResult(
                query=query,
                intent=intent,
                candidates=[],
                selected_memories=[],
                total_tokens_used=0,
                token_budget=budget,
                budget_compliant=True,
                lineage_chains=lineage_chains or {},
            )

        selected_memories: List[MemoryRecord] = []
        selected_tokens = 0
        selected_words: Set[str] = set()

        # Step 1: Compute initial baseline utility for each candidate
        for c in candidates:
            # Baseline utility without redundancy penalty
            c.utility_score = (
                self.w_rel * c.relevance_score
                + self.w_temp * c.temporal_score
                + self.w_imp * c.importance_score
            )

        # Step 2: Sort candidates descending by baseline utility
        candidates.sort(key=lambda x: x.utility_score, reverse=True)

        # Step 3: Greedy selection with dynamic redundancy penalty
        for c in candidates:
            cand_tokens = c.estimated_tokens or self._estimate_tokens(c.memory)

            # Check if adding this candidate fits in budget
            if selected_tokens + cand_tokens <= budget:
                # Calculate redundancy against already selected memories
                cand_words = set(f"{c.memory.subject} {c.memory.attribute} {c.memory.value}".lower().split())
                overlap = len(cand_words.intersection(selected_words))
                redundancy_penalty = (overlap / max(len(cand_words), 1)) if cand_words else 0.0

                c.redundancy_penalty = redundancy_penalty
                c.utility_score = max(0.0, c.utility_score - (self.w_red * redundancy_penalty))

                # Mark selected
                c.selected = True
                selected_memories.append(c.memory)
                selected_tokens += cand_tokens
                selected_words.update(cand_words)
            else:
                c.selected = False

        logger.info(
            f"Context budget optimization: selected {len(selected_memories)}/{len(candidates)} memories. "
            f"Tokens: {selected_tokens}/{budget} (compliant: {selected_tokens <= budget})"
        )

        return RetrievalResult(
            query=query,
            intent=intent,
            candidates=candidates,
            selected_memories=selected_memories,
            total_tokens_used=selected_tokens,
            token_budget=budget,
            budget_compliant=selected_tokens <= budget,
            lineage_chains=lineage_chains or {},
        )

    def _estimate_tokens(self, record: MemoryRecord) -> int:
        """Estimate token count for a memory item."""
        text = f"{record.subject} {record.attribute}: {record.value} ({record.memory_type.value})"
        return max(len(text) // 4, 1)
