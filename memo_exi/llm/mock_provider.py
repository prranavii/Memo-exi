"""
Deterministic Mock and Heuristic LLM provider for testing and offline execution.
Produces valid JSON and grounded responses following prompt instructions.
"""

from datetime import datetime, timezone
import json
import logging
import re
from typing import Optional
from memo_exi.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class MockLLMProvider(LLMProvider):
    """Deterministic offline provider executing heuristic rule-based extraction and responses."""

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        prompt_lower = prompt.lower()
        sys_lower = (system_prompt or "").lower()

        # 1. Fact Extraction request
        if "extract" in sys_lower or "extract meaningful facts" in prompt_lower or "json_schema" in prompt_lower:
            return self._heuristic_fact_extraction(prompt)

        # 2. Query classification request
        if "classify" in sys_lower or "query intent" in prompt_lower:
            return self._heuristic_query_classification(prompt)

        # 3. Conflict resolution request
        if "conflict" in sys_lower or "contradiction" in prompt_lower:
            return self._heuristic_conflict_resolution(prompt)

        # 4. Response generation
        return self._heuristic_response_generation(prompt)

    def _heuristic_fact_extraction(self, prompt: str) -> str:
        """Heuristic rule-based fact extractor for offline operation and tests."""
        facts = []
        now_iso = datetime.now(timezone.utc).isoformat()

        # Chitchat filter
        chitchat_phrases = ["hello", "hi there", "how are you", "good morning", "good evening", "thanks", "thank you"]
        cleaned_prompt = prompt.strip()
        for phrase in chitchat_phrases:
            if cleaned_prompt.lower().startswith(phrase) and len(cleaned_prompt.split()) <= 4:
                return json.dumps({"facts": [], "filtered_out_chitchat": True})

        # Pattern 1: Salary extraction
        salary_match = re.search(r"(?:salary|annual income|compensation)(?:\s+(?:is|became|increased to|was|now))\s+(\$?\d+(?:,\d+)*(?:\s*(?:k|usd|per year))?)", prompt, re.IGNORECASE)
        if salary_match:
            val = salary_match.group(1).strip()
            facts.append({
                "subject": "User",
                "attribute": "salary",
                "value": val,
                "memory_type": "factual",
                "confidence_score": 0.95,
                "importance_score": 0.85,
                "valid_from": now_iso,
                "source_provenance": "user_statement",
                "raw_snippet": salary_match.group(0),
            })

        # Pattern 2: Role / title
        role_match = re.search(r"(?:promoted to|working as|role is|job title is|position is)\s+([A-Za-z\s\-]+?)(?:\.|\n|$|,|\s+at|\s+starting)", prompt, re.IGNORECASE)
        if role_match:
            facts.append({
                "subject": "User",
                "attribute": "role",
                "value": role_match.group(1).strip(),
                "memory_type": "factual",
                "confidence_score": 0.95,
                "importance_score": 0.8,
                "valid_from": now_iso,
                "source_provenance": "user_statement",
                "raw_snippet": role_match.group(0),
            })

        # Pattern 3: Deadline
        deadline_match = re.search(r"(?:deadline|launch date|milestone)\s+(?:for\s+([A-Za-z0-9_\-\s]+?)\s+)?(?:is|was|moved to|postponed to|changed to)\s+([A-Za-z0-9\-/\s]+?)(?:\.|\n|$|,)", prompt, re.IGNORECASE)
        if deadline_match:
            proj = (deadline_match.group(1) or "Project").strip()
            val = deadline_match.group(2).strip()
            facts.append({
                "subject": proj,
                "attribute": "deadline",
                "value": val,
                "memory_type": "factual",
                "confidence_score": 0.92,
                "importance_score": 0.9,
                "valid_from": now_iso,
                "source_provenance": "user_statement",
                "raw_snippet": deadline_match.group(0),
            })

        # Pattern 4: Preferences
        pref_match = re.search(r"(?:i\s+(?:prefer|love|like|enjoy|always use))\s+([^.\n]+)", prompt, re.IGNORECASE)
        if pref_match and not any(f["attribute"] in ["salary", "role", "deadline"] for f in facts):
            facts.append({
                "subject": "User",
                "attribute": "preference",
                "value": pref_match.group(1).strip(),
                "memory_type": "preference",
                "confidence_score": 0.9,
                "importance_score": 0.7,
                "valid_from": now_iso,
                "source_provenance": "user_statement",
                "raw_snippet": pref_match.group(0),
            })

        # Pattern 5: Constraints
        constraint_match = re.search(r"(?:never|always|do not|must not|ensure that)\s+([^.\n]+)", prompt, re.IGNORECASE)
        if constraint_match and not facts:
            facts.append({
                "subject": "User",
                "attribute": "constraint",
                "value": constraint_match.group(0).strip(),
                "memory_type": "constraint",
                "confidence_score": 0.95,
                "importance_score": 0.85,
                "valid_from": now_iso,
                "source_provenance": "user_statement",
                "raw_snippet": constraint_match.group(0),
            })

        # Fallback generic subject-verb-object if still empty and informative
        if not facts and len(prompt.split()) > 3 and not any(p in prompt.lower() for p in chitchat_phrases):
            facts.append({
                "subject": "User",
                "attribute": "general_statement",
                "value": prompt.strip(),
                "memory_type": "episodic",
                "confidence_score": 0.8,
                "importance_score": 0.5,
                "valid_from": now_iso,
                "source_provenance": "user_interaction",
                "raw_snippet": prompt.strip(),
            })

        return json.dumps({"facts": facts, "filtered_out_chitchat": len(facts) == 0})

    def _heuristic_query_classification(self, prompt: str) -> str:
        """Heuristic rule-based query classifier."""
        p_lower = prompt.lower()
        now_year = datetime.now(timezone.utc).year

        # 1. Historical questions: past tense, specific dates, "before", "in 2024", "was"
        if any(w in p_lower for w in ["what was", "where did i live", "before", "in 2023", "in 2024", "prior to", "previously", "originally"]):
            # Extract date if mentioned
            date_match = re.search(r"(?:before|in|prior to|as of)\s+([A-Za-z]+\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{4})", prompt, re.IGNORECASE)
            target_date = date_match.group(1) if date_match else "2024-01-01"
            return json.dumps({
                "intent": "historical",
                "time_constraint": target_date,
                "subject": "User",
                "attribute": "salary" if "salary" in p_lower else "attribute",
                "explanation": "Query seeks historical state prior to a temporal boundary.",
            })

        # 2. Temporal evolution / lineage: "when did", "how many times", "history of", "how has ... changed"
        if any(w in p_lower for w in ["when did", "how many times", "how has", "history of", "track the changes", "evolution"]):
            return json.dumps({
                "intent": "temporal",
                "subject": "User",
                "attribute": "deadline" if "deadline" in p_lower else "salary" if "salary" in p_lower else "attribute",
                "explanation": "Query asks for change timestamps or lineage history.",
            })

        # 3. Current-state questions: "what is my current", "what is my salary now", "where do i currently"
        if any(w in p_lower for w in ["current", "currently", "now", "today", "latest", "what is my"]):
            return json.dumps({
                "intent": "current_state",
                "subject": "User",
                "attribute": "salary" if "salary" in p_lower else "attribute",
                "explanation": "Query requests the active current fact.",
            })

        # 4. Ambiguous queries: very short, under-specified
        if len(p_lower.split()) <= 2 and "?" in p_lower:
            return json.dumps({
                "intent": "ambiguous",
                "explanation": "Query lacks sufficient entity and attribute specification.",
            })

        # 5. Default: general semantic
        return json.dumps({
            "intent": "general_semantic",
            "explanation": "General topical or semantic recall question.",
        })

    def _heuristic_conflict_resolution(self, prompt: str) -> str:
        """Heuristic conflict analyzer."""
        return json.dumps({
            "is_contradiction": False,
            "is_valid_update": True,
            "requires_clarification": False,
            "explanation": "Heuristic resolution: detected regular fact update.",
        })

    def _heuristic_response_generation(self, prompt: str) -> str:
        """Grounded response synthesis from provided memories."""
        return "Based on your verified memory records, your current information is documented and preserved."
