"""
Module E: Conflict-Aware Resolution.
Detects contradictory memory updates, evaluates confidence deltas and provenance,
and initiates clarification workflows for unresolved disputes.
"""

from datetime import datetime
import json
import logging
import re
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from memo_exi.config import settings
from memo_exi.llm.base import LLMProvider
from memo_exi.llm.factory import get_llm_provider
from memo_exi.models.memory_models import ExtractedFact, MemoryRecord

logger = logging.getLogger(__name__)

SYSTEM_CONFLICT_PROMPT = """You are an expert Conflict and Contradiction Resolution Engine for an AI Agent.
Your task is to analyze an EXISTING ACTIVE MEMORY and an INCOMING CANDIDATE FACT to determine if they are:
1. A LEGITIMATE TEMPORAL UPDATE (e.g. user moved, was promoted, salary increased, deadline postponed).
2. A TRUE CONTRADICTION / DISPUTE (e.g. simultaneously asserting mutually exclusive facts without indicating a change).
3. A DUPLICATE or compatible refinement.

CRITERIA:
- If the incoming fact contains transition language ("now", "moved to", "promoted", "changed", "updated", "effective"), it is usually a legitimate update.
- If the two assertions are logically incompatible (e.g. "I am vegan" vs "I eat beef daily", "I live exclusively in Berlin" vs "I live exclusively in Tokyo") and no transition was stated, it is a contradiction requiring clarification.

OUTPUT FORMAT (JSON strictly):
{
  "is_contradiction": true | false,
  "is_legitimate_update": true | false,
  "requires_clarification": true | false,
  "clarification_question": "Natural clarifying question to present to user, or null",
  "explanation": "Brief rationale"
}
"""


class ConflictAnalysisResult(BaseModel):
    """Evaluation of potential contradiction between existing memory and incoming fact."""
    is_contradiction: bool = False
    is_legitimate_update: bool = True
    requires_clarification: bool = False
    clarification_question: Optional[str] = None
    confidence_delta: float = 0.0
    explanation: Optional[str] = None


class ConflictResolver:
    """Analyzes memory updates for contradictions and manages clarification requests."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def analyze_update(
        self,
        existing: MemoryRecord,
        incoming: ExtractedFact,
    ) -> ConflictAnalysisResult:
        """
        Evaluate if incoming fact contradicts existing memory or is a valid version update.
        """
        # Exact/case-insensitive match is a duplicate, not a contradiction
        if existing.value.strip().lower() == incoming.value.strip().lower():
            return ConflictAnalysisResult(
                is_contradiction=False,
                is_legitimate_update=False,
                requires_clarification=False,
                explanation="Duplicate value.",
            )

        # Transition cues in raw context strongly signify a valid chronological update
        transition_cues = [
            "now", "moved to", "promoted to", "changed to", "became", "updated to",
            "increased to", "new", "effective", "starting", "postponed to", "rescheduled to",
        ]
        context = (incoming.raw_snippet or "").lower()
        has_transition_cue = any(cue in context for cue in transition_cues)

        # Confidence delta
        conf_delta = abs(incoming.confidence_score - existing.confidence_score)

        if has_transition_cue:
            return ConflictAnalysisResult(
                is_contradiction=False,
                is_legitimate_update=True,
                requires_clarification=False,
                confidence_delta=conf_delta,
                explanation="Detected explicit chronological transition cue.",
            )

        # Check for opposing categorical contradictions using LLM / heuristic
        prompt = (
            f"Existing Active Memory:\n"
            f"- Subject: {existing.subject}\n"
            f"- Attribute: {existing.attribute}\n"
            f"- Value: {existing.value}\n"
            f"- Confidence: {existing.confidence_score}\n"
            f"- Recorded at: {existing.valid_from.isoformat()}\n\n"
            f"Incoming Candidate Fact:\n"
            f"- Value: {incoming.value}\n"
            f"- Confidence: {incoming.confidence_score}\n"
            f"- Context: \"{incoming.raw_snippet or ''}\"\n"
        )

        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system_prompt=SYSTEM_CONFLICT_PROMPT,
                json_mode=True,
            )
            parsed = self._parse_json(raw_response)
        except Exception as e:
            logger.warning(f"LLM conflict analysis failed: {e}. Using heuristic.")
            parsed = self._heuristic_fallback(existing, incoming)

        is_contra = parsed.get("is_contradiction", False)
        requires_clarif = parsed.get("requires_clarification", is_contra)
        clarif_q = parsed.get("clarification_question")

        if is_contra and not clarif_q:
            clarif_q = (
                f"I noticed a potential conflict regarding your {existing.attribute}. "
                f"Previously you indicated '{existing.value}', but now you mentioned '{incoming.value}'. "
                f"Which one is currently accurate?"
            )

        return ConflictAnalysisResult(
            is_contradiction=is_contra,
            is_legitimate_update=parsed.get("is_legitimate_update", not is_contra),
            requires_clarification=requires_clarif,
            clarification_question=clarif_q,
            confidence_delta=conf_delta,
            explanation=parsed.get("explanation"),
        )

    def _heuristic_fallback(
        self,
        existing: MemoryRecord,
        incoming: ExtractedFact,
    ) -> Dict[str, Any]:
        """Heuristic check for opposing traits or identical attributes without cues."""
        val_old = existing.value.lower()
        val_new = incoming.value.lower()

        # Check for opposing words
        opposing_pairs = [
            ("vegetarian", "meat"), ("vegan", "steak"), ("mac", "windows"),
            ("remote", "in-office"), ("single", "married"),
        ]
        for a, b in opposing_pairs:
            if (a in val_old and b in val_new) or (b in val_old and a in val_new):
                return {
                    "is_contradiction": True,
                    "is_legitimate_update": False,
                    "requires_clarification": True,
                    "explanation": f"Detected diametrically opposing assertions: '{existing.value}' vs '{incoming.value}'.",
                }

        # Otherwise treat as normal update
        return {
            "is_contradiction": False,
            "is_legitimate_update": True,
            "requires_clarification": False,
            "explanation": "Treated as state update.",
        }

    def _parse_json(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]
        return json.loads(cleaned)
