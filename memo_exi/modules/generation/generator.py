"""
Module H: LLM Response Generation.
Synthesizes grounded, temporally faithful responses based on retrieved memories.
Explicitly delineates active current facts, historical states, disputed facts, and missing data.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from memo_exi.llm.base import LLMProvider
from memo_exi.llm.factory import get_llm_provider
from memo_exi.models.memory_models import (
    MemoryRecord,
    MemoryStatus,
    QueryIntent,
    RetrievalResult,
    get_current_utc,
)

logger = logging.getLogger(__name__)

GROUNDED_SYSTEM_PROMPT = """You are MEMO-EXI, an AI Assistant equipped with an adaptive temporal memory engine.
Your answers MUST be strictly grounded in the verified memory records provided below.

CRITICAL TEMPORAL RULES:
1. NEVER present superseded or outdated information as the user's current state.
2. If the user asks a current-state question, cite ONLY memories marked [CURRENT].
3. If the user asks a historical or retrospective question, explain what the fact was during that specific timeframe based on [HISTORICAL] records.
4. If a fact is marked [DISPUTED], inform the user that conflicting information exists and ask for clarification.
5. If the requested information is absent from the records, clearly state that you have no verified memory of it. Do not fabricate or extrapolate.
6. Provide concise, clear, and direct answers.
"""


class GeneratedResponse(BaseModel):
    """The synthesized answer along with grounding provenance metadata."""
    answer: str
    memories_cited: List[str] = Field(default_factory=list)
    has_disputed_facts: bool = False
    grounded: bool = True


class ResponseGenerator:
    """Generates grounded responses based on temporal memories."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def generate_response(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        reference_time: Optional[datetime] = None,
    ) -> GeneratedResponse:
        """
        Synthesize answer from retrieved memories respecting temporal status and query intent.
        """
        ref_time = reference_time or get_current_utc()
        memories = retrieval_result.selected_memories
        lineage = retrieval_result.lineage_chains

        # If zero memories retrieved
        if not memories and not lineage:
            return GeneratedResponse(
                answer="I do not have any recorded memory regarding this information.",
                memories_cited=[],
                has_disputed_facts=False,
                grounded=True,
            )

        # Build structured memory context
        current_facts = []
        historical_facts = []
        disputed_facts = []

        for m in memories:
            valid_from_str = m.valid_from.strftime("%Y-%m-%d")
            valid_until_str = m.valid_until.strftime("%Y-%m-%d") if m.valid_until else "present"
            entry = f"- {m.subject} {m.attribute}: \"{m.value}\" [Valid: {valid_from_str} to {valid_until_str}] (type: {m.memory_type.value})"

            if m.status == MemoryStatus.CURRENT:
                current_facts.append(f"[CURRENT] {entry}")
            elif m.status == MemoryStatus.SUPERSEDED:
                historical_facts.append(f"[HISTORICAL (SUPERSEDED)] {entry}")
            elif m.status == MemoryStatus.DISPUTED:
                disputed_facts.append(f"[DISPUTED] {entry}")

        context_blocks = []
        if current_facts:
            context_blocks.append("### ACTIVE CURRENT FACTS:\n" + "\n".join(current_facts))
        if historical_facts:
            context_blocks.append("### HISTORICAL MEMORIES (Prior States):\n" + "\n".join(historical_facts))
        if disputed_facts:
            context_blocks.append("### DISPUTED FACTS (Pending Clarification):\n" + "\n".join(disputed_facts))

        if lineage:
            lineage_blocks = []
            for k, chain in lineage.items():
                items_str = " -> ".join([f"'{v.value}' ({v.valid_from.strftime('%Y-%m')})" for v in chain])
                lineage_blocks.append(f"- {k}: {items_str}")
            context_blocks.append("### FACT EVOLUTION TIMELINES:\n" + "\n".join(lineage_blocks))

        memory_context_str = "\n\n".join(context_blocks)

        user_prompt = (
            f"Current As-Of Timestamp: {ref_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Query Intent Detected: {retrieval_result.intent.value}\n\n"
            f"VERIFIED MEMORY RECORDS:\n"
            f"{memory_context_str}\n\n"
            f"User Question: \"{query}\"\n\n"
            f"Answer the question directly, strictly adhering to the temporal rules."
        )

        try:
            raw_answer = self.llm.generate(
                prompt=user_prompt,
                system_prompt=GROUNDED_SYSTEM_PROMPT,
            )
            answer_text = raw_answer.strip()
        except Exception as e:
            logger.error(f"Response generation failed: {e}. Using deterministic fallback.")
            answer_text = self._deterministic_fallback_answer(query, retrieval_result)

        cited_ids = [m.memory_id for m in memories]
        return GeneratedResponse(
            answer=answer_text,
            memories_cited=cited_ids,
            has_disputed_facts=len(disputed_facts) > 0,
            grounded=True,
        )

    def _deterministic_fallback_answer(
        self, query: str, retrieval_result: RetrievalResult
    ) -> str:
        """Rule-based fallback when LLM is unavailable."""
        intent = retrieval_result.intent
        memories = retrieval_result.selected_memories

        if not memories:
            return "I do not have any recorded memory regarding this in my database."

        if intent == QueryIntent.CURRENT_STATE:
            curr = [m for m in memories if m.status == MemoryStatus.CURRENT]
            if curr:
                return f"According to your current records, your {curr[0].attribute} is {curr[0].value}."
            return "There is no currently active record for this."

        if intent == QueryIntent.HISTORICAL:
            hist = [m for m in memories if m.status == MemoryStatus.SUPERSEDED]
            if hist:
                return f"Historically, your {hist[0].attribute} was {hist[0].value} (valid until {hist[0].valid_until.strftime('%Y-%m-%d') if hist[0].valid_until else 'N/A'})."
            if memories:
                return f"Recorded as {memories[0].value}."

        if intent == QueryIntent.TEMPORAL and retrieval_result.lineage_chains:
            for k, chain in retrieval_result.lineage_chains.items():
                steps = " then ".join([f"'{c.value}'" for c in chain])
                return f"The history of {k.replace('::', ' ')} progressed as: {steps}."

        return f"Based on your records: {memories[0].subject} {memories[0].attribute} is {memories[0].value}."
