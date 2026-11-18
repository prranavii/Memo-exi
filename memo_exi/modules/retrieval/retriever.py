"""
Module D: Semantic Memory Retrieval.
Executes vector similarity search in ChromaDB, applies temporal interval and status filters,
and enriches results with authoritative SQLite metadata.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from memo_exi.config import settings
from memo_exi.database.sqlite_manager import SQLiteMemoryManager
from memo_exi.models.memory_models import (
    MemoryRecord,
    MemoryStatus,
    QueryIntent,
    RetrievalCandidate,
)
from memo_exi.modules.classification.classifier import QueryClassificationResult
from memo_exi.vector_store.chroma_manager import ChromaMemoryManager

logger = logging.getLogger(__name__)


class SemanticTemporalRetriever:
    """Hybrid semantic and temporal memory retrieval engine."""

    def __init__(
        self,
        sqlite_manager: Optional[SQLiteMemoryManager] = None,
        chroma_manager: Optional[ChromaMemoryManager] = None,
    ):
        self.sqlite = sqlite_manager or SQLiteMemoryManager()
        self.chroma = chroma_manager or ChromaMemoryManager()

    def retrieve(
        self,
        query: str,
        classification: QueryClassificationResult,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievalCandidate]:
        """
        Execute routed retrieval based on query classification.
        Returns candidate memories enriched with relevance and temporal validity scores.
        """
        k = top_k or settings.default_top_k
        intent = classification.intent

        # Step 1: Determine metadata filters for vector query
        where_filter: Optional[Dict[str, Any]] = None
        if session_id:
            where_filter = {"session_id": session_id}

        # If strict current-state query, filter vector search by status='current'
        if intent == QueryIntent.CURRENT_STATE:
            if where_filter:
                where_filter = {"$and": [{"session_id": session_id}, {"status": "current"}]}
            else:
                where_filter = {"status": "current"}

        # Step 2: Query ChromaDB for candidate memory IDs
        # Fetch slightly more candidates (2x k) to allow post-filtering by temporal constraints
        fetch_k = max(k * 2, 10)
        vector_matches = self.chroma.query_similar(
            query_text=query,
            top_k=fetch_k,
            where=where_filter,
        )

        if not vector_matches:
            # Fallback: if vector search yielded zero, try direct SQLite subject/attribute search
            return self._fallback_sqlite_search(classification, session_id, k)

        # Step 3: Fetch authoritative records from SQLite & evaluate temporal suitability
        candidates: List[RetrievalCandidate] = []
        for match in vector_matches:
            mem_id = match["memory_id"]
            record = self.sqlite.get_memory_by_id(mem_id)
            if not record:
                continue

            similarity = match["similarity"]
            temp_score = self._compute_temporal_suitability(record, classification)

            # Strict filter enforcement based on intent:
            # For current_state query, exclude superseded or disputed memories
            if intent == QueryIntent.CURRENT_STATE and record.status != MemoryStatus.CURRENT:
                continue

            # For historical query with target_datetime, verify valid interval
            if intent == QueryIntent.HISTORICAL and classification.target_datetime:
                t = classification.target_datetime
                # Must be valid at time t: valid_from <= t AND (valid_until IS NULL OR valid_until > t)
                if not (record.valid_from <= t and (record.valid_until is None or record.valid_until > t)):
                    temp_score *= 0.1  # Heavy penalty if outside temporal window

            candidate = RetrievalCandidate(
                memory=record,
                relevance_score=similarity,
                temporal_score=temp_score,
                importance_score=record.importance_score,
                redundancy_penalty=0.0,
                utility_score=0.0,  # Computed in Module F (Context-Budget Manager)
                estimated_tokens=self._estimate_tokens(record),
                selected=False,
            )
            candidates.append(candidate)

        return candidates[:k]

    def retrieve_lineage_chains(
        self,
        candidates: List[RetrievalCandidate],
    ) -> Dict[str, List[MemoryRecord]]:
        """For temporal evolution queries, fetch full lineage chains for each unique entity-attribute."""
        chains: Dict[str, List[MemoryRecord]] = {}
        seen_keys = set()

        for cand in candidates:
            key = f"{cand.memory.subject}::{cand.memory.attribute}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            history = self.sqlite.get_fact_history(
                subject=cand.memory.subject,
                attribute=cand.memory.attribute,
                session_id=cand.memory.session_id,
            )
            if history:
                chains[key] = history

        return chains

    def _compute_temporal_suitability(
        self,
        record: MemoryRecord,
        classification: QueryClassificationResult,
    ) -> float:
        """Score how well a memory aligns with the temporal constraints of the query [0.0 - 1.0]."""
        intent = classification.intent

        if intent == QueryIntent.CURRENT_STATE:
            return 1.0 if record.status == MemoryStatus.CURRENT else 0.0

        if intent == QueryIntent.HISTORICAL and classification.target_datetime:
            t = classification.target_datetime
            if record.valid_from <= t and (record.valid_until is None or record.valid_until > t):
                return 1.0
            return 0.1

        if intent == QueryIntent.TEMPORAL:
            # For change tracking, all versions in lineage have high temporal utility
            return 0.9

        # General semantic query: slight preference for current facts
        if record.status == MemoryStatus.CURRENT:
            return 1.0
        elif record.status == MemoryStatus.SUPERSEDED:
            return 0.5
        return 0.2

    def _estimate_tokens(self, record: MemoryRecord) -> int:
        """Estimate token count for prompt budget packing."""
        text = f"{record.subject} {record.attribute}: {record.value} (valid: {record.valid_from.date()})"
        # Rule of thumb: ~4 characters per token
        return max(len(text) // 4, 1)

    def _fallback_sqlite_search(
        self,
        classification: QueryClassificationResult,
        session_id: Optional[str],
        k: int,
    ) -> List[RetrievalCandidate]:
        """Fallback to SQLite querying if vector store returns empty."""
        records = self.sqlite.query_memories(
            session_id=session_id,
            subject=classification.subject,
            attribute=classification.attribute,
            as_of_time=classification.target_datetime if classification.intent == QueryIntent.HISTORICAL else None,
            status=MemoryStatus.CURRENT if classification.intent == QueryIntent.CURRENT_STATE else None,
            limit=k,
        )
        return [
            RetrievalCandidate(
                memory=rec,
                relevance_score=0.8,
                temporal_score=1.0,
                importance_score=rec.importance_score,
                estimated_tokens=self._estimate_tokens(rec),
            )
            for rec in records
        ]
