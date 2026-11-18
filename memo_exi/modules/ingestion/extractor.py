"""
Module A: Conversation Ingestion and Structured Fact Extraction.
Extracts structured factual, preference, constraint, and episodic memories from natural language.
Filters non-informational chitchat and computes confidence and importance scores.
"""

from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, Dict, List, Optional
from memo_exi.llm.base import LLMProvider
from memo_exi.llm.factory import get_llm_provider
from memo_exi.models.memory_models import (
    ExtractedFact,
    ExtractionResult,
    MemoryType,
    get_current_utc,
)

logger = logging.getLogger(__name__)

SYSTEM_EXTRACTION_PROMPT = """You are an expert Knowledge Extraction and Memory Ingestion Engine for an AI Agent.
Your task is to analyze user conversation turns and extract meaningful, persistent facts into a structured JSON schema.

RULES:
1. Extract ONLY concrete, meaningful, long-term facts, preferences, constraints, or episodic events.
2. DO NOT extract transient conversational filler, greetings, pleasantries, questions, or temporary chit-chat (e.g., "Hello", "How are you?", "Thanks!").
   If the message contains no persistent facts, return an empty "facts" array with "filtered_out_chitchat": true.
3. Classify each memory into exactly one "memory_type":
   - "preference": Personal likes, dislikes, habits, style choices (e.g. "I prefer dark mode", "I like Python").
   - "factual": Definite attributes, job titles, salaries, locations, version numbers (e.g. "My salary is $120,000", "I moved to Seattle").
   - "constraint": Explicit rules, prohibitions, or guidelines (e.g. "Never use tabs", "Always write type hints").
   - "episodic": Events tied to specific occurrences (e.g. "We launched release v2.0 yesterday").
4. Assign a "confidence_score" between 0.0 and 1.0 (how certain is this factual claim from the text).
5. Assign an "importance_score" between 0.0 and 1.0 (how critical is this fact for future agent decisions).
6. Anchor temporal statements: If the user specifies an effective date (e.g. "effective Oct 2024", "starting next month"), provide it in ISO 8601 format in "valid_from". If not specified, leave as null.
7. Output strictly valid JSON with this exact format:
{
  "facts": [
    {
      "subject": "Entity name (e.g., User, Project, Team)",
      "attribute": "Attribute name in snake_case or concise phrase (e.g., salary, location, preferred_editor)",
      "value": "The factual value (e.g., $120,000, Seattle, VS Code)",
      "memory_type": "preference" | "factual" | "constraint" | "episodic",
      "confidence_score": 0.95,
      "importance_score": 0.8,
      "valid_from": null,
      "source_provenance": "user_statement",
      "raw_snippet": "Exact snippet from the text"
    }
  ],
  "filtered_out_chitchat": false
}
"""


class FactExtractor:
    """Extracts structured memory records from conversational turns."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def extract_facts(
        self,
        text: str,
        session_id: str = "default_session",
        reference_time: Optional[datetime] = None,
    ) -> ExtractionResult:
        """
        Extract facts from conversational message.
        Validates against Pydantic models.
        """
        ref_time = reference_time or get_current_utc()
        cleaned = text.strip()
        if not cleaned:
            return ExtractionResult(
                session_id=session_id,
                facts=[],
                raw_message=text,
                filtered_out_chitchat=True,
            )

        prompt = f"Reference timestamp: {ref_time.isoformat()}\n\nUser message:\n\"\"\"{cleaned}\"\"\""

        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system_prompt=SYSTEM_EXTRACTION_PROMPT,
                json_mode=True,
            )
            parsed_data = self._parse_json_response(raw_response)
        except Exception as e:
            logger.error(f"LLM fact extraction failed: {e}. Falling back to empty extraction.", exc_info=True)
            parsed_data = {"facts": [], "filtered_out_chitchat": True}

        raw_facts = parsed_data.get("facts", [])
        filtered_chitchat = parsed_data.get("filtered_out_chitchat", len(raw_facts) == 0)

        validated_facts: List[ExtractedFact] = []
        for item in raw_facts:
            try:
                # Ensure memory_type matches enum
                mem_type_str = item.get("memory_type", "factual").lower()
                if mem_type_str not in [m.value for m in MemoryType]:
                    mem_type_str = "factual"

                valid_from_val = None
                if item.get("valid_from"):
                    try:
                        valid_from_val = datetime.fromisoformat(item["valid_from"])
                    except Exception:
                        valid_from_val = ref_time
                else:
                    valid_from_val = ref_time

                fact = ExtractedFact(
                    subject=item.get("subject", "User").strip(),
                    attribute=item.get("attribute", "general").strip().lower(),
                    value=str(item.get("value", "")).strip(),
                    memory_type=MemoryType(mem_type_str),
                    confidence_score=float(item.get("confidence_score", 0.9)),
                    importance_score=float(item.get("importance_score", 0.5)),
                    valid_from=valid_from_val,
                    source_provenance=item.get("source_provenance", "user_statement"),
                    raw_snippet=item.get("raw_snippet", cleaned),
                )
                if fact.value:
                    validated_facts.append(fact)
            except Exception as e:
                logger.warning(f"Skipping malformed extracted fact {item}: {e}")

        return ExtractionResult(
            session_id=session_id,
            facts=validated_facts,
            raw_message=text,
            filtered_out_chitchat=filtered_chitchat or len(validated_facts) == 0,
        )

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Robustly parse JSON from LLM response, stripping markdown fences if present."""
        cleaned = text.strip()
        # Remove ```json and ``` fences
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Find first { and last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]

        return json.loads(cleaned)
