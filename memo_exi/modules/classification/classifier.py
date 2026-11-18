"""
Module C: Temporal Query Classifier.
Classifies incoming questions into 5 intent categories and extracts temporal constraints.
"""

from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from memo_exi.llm.base import LLMProvider
from memo_exi.llm.factory import get_llm_provider
from memo_exi.models.memory_models import QueryIntent, get_current_utc

logger = logging.getLogger(__name__)

SYSTEM_CLASSIFICATION_PROMPT = """You are an expert Temporal Query Classifier for an intelligent memory engine.
Your task is to classify user queries into one of 5 distinct categories, and extract temporal and entity constraints.

CATEGORIES:
1. "current_state": The user is asking about an active, present fact (e.g. "What is my current salary?", "Where do I live?", "Who is my manager now?").
2. "historical": The user is asking about past information anchored to a specific time or condition before an update (e.g. "What was my salary before October?", "Where did I live in 2022?", "What was the original deadline?").
3. "temporal": The user is asking about fact changes, evolution, history, or change timestamps (e.g. "When did my project deadline change?", "How many times was my salary updated?", "Show the history of my job titles").
4. "general_semantic": General or semantic recall question not tied to a specific point in time (e.g. "What are my preferences for coding?", "Tell me about Project Alpha").
5. "ambiguous": The query is ambiguous, missing entities, or too vague (e.g. "What about it?", "Why?", "What was that?").

OUTPUT SCHEMA:
Provide valid JSON matching this schema:
{
  "intent": "current_state" | "historical" | "temporal" | "general_semantic" | "ambiguous",
  "time_constraint": "ISO date string or descriptive date anchor like '2024-10-01' or null",
  "subject": "Entity name or null",
  "attribute": "Attribute name or null",
  "explanation": "Brief rationale"
}
"""


class QueryClassificationResult(BaseModel):
    """Result of intent classification and constraint extraction."""
    intent: QueryIntent
    raw_query: str
    target_datetime: Optional[datetime] = None
    time_constraint_str: Optional[str] = None
    subject: Optional[str] = None
    attribute: Optional[str] = None
    explanation: Optional[str] = None


class TemporalQueryClassifier:
    """Classifies user queries and parses point-in-time constraints."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def classify_query(
        self,
        query: str,
        reference_time: Optional[datetime] = None,
    ) -> QueryClassificationResult:
        """Classify query intent and extract point-in-time constraints."""
        ref_time = reference_time or get_current_utc()
        cleaned = query.strip()

        # Handle trivial / ambiguous queries directly
        if len(cleaned.split()) <= 2 and "?" in cleaned:
            return QueryClassificationResult(
                intent=QueryIntent.AMBIGUOUS,
                raw_query=query,
                explanation="Short ambiguous query without entity context.",
            )

        prompt = f"Reference timestamp: {ref_time.isoformat()}\n\nUser Query: \"{cleaned}\""

        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system_prompt=SYSTEM_CLASSIFICATION_PROMPT,
                json_mode=True,
            )
            parsed = self._parse_json(raw_response)
        except Exception as e:
            logger.warning(f"LLM query classification failed: {e}. Using rule fallback.")
            parsed = self._rule_based_fallback(cleaned)

        intent_str = parsed.get("intent", "general_semantic").lower()
        if intent_str not in [i.value for i in QueryIntent]:
            intent_str = "general_semantic"
        intent = QueryIntent(intent_str)

        time_str = parsed.get("time_constraint")
        target_dt = self._parse_date_constraint(time_str, cleaned, ref_time)

        return QueryClassificationResult(
            intent=intent,
            raw_query=query,
            target_datetime=target_dt,
            time_constraint_str=time_str,
            subject=parsed.get("subject"),
            attribute=parsed.get("attribute"),
            explanation=parsed.get("explanation"),
        )

    def _parse_date_constraint(
        self,
        time_str: Optional[str],
        query: str,
        ref_time: datetime,
    ) -> Optional[datetime]:
        """Convert extracted time string or text mention to timezone-aware datetime."""
        # 1. Try ISO parsing from model output
        if time_str:
            try:
                dt = datetime.fromisoformat(time_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass

        # 2. Heuristic extraction of common months / years
        month_map = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10,
            "november": 11, "nov": 11, "december": 12, "dec": 12,
        }

        q_lower = query.lower()
        # Look for e.g. "before October 2024" or "before October"
        match = re.search(
            r"(?:before|in|prior to|until|as of)\s+([A-Za-z]+)(?:\s+(\d{4}))?",
            q_lower,
        )
        if match:
            month_name = match.group(1).lower()
            year_val = int(match.group(2)) if match.group(2) else ref_time.year
            if month_name in month_map:
                month_num = month_map[month_name]
                return datetime(year_val, month_num, 1, 0, 0, tzinfo=timezone.utc)

        # Look for explicit year e.g. "in 2023"
        year_match = re.search(r"\b(20\d\d)\b", query)
        if year_match:
            year_val = int(year_match.group(1))
            return datetime(year_val, 1, 1, 0, 0, tzinfo=timezone.utc)

        return None

    def _rule_based_fallback(self, query: str) -> Dict[str, Any]:
        """Deterministic rule fallback when LLM is unavailable."""
        q_lower = query.lower()
        if any(w in q_lower for w in ["what was", "before", "previously", "originally", "prior to"]):
            return {"intent": "historical"}
        if any(w in q_lower for w in ["when did", "how many times", "how has", "history of"]):
            return {"intent": "temporal"}
        if any(w in q_lower for w in ["current", "currently", "now", "today", "latest", "what is my"]):
            return {"intent": "current_state"}
        return {"intent": "general_semantic"}

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
