"""
Module B: Version-Aware Temporal Memory Manager.
Handles non-destructive fact updates, duplicate suppression, superseding transactions,
lineage tracking, and dual-store SQLite + ChromaDB synchronization.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from memo_exi.database.sqlite_manager import SQLiteMemoryManager
from memo_exi.models.memory_models import (
    ExtractedFact,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    get_current_utc,
)
from memo_exi.vector_store.chroma_manager import ChromaMemoryManager

logger = logging.getLogger(__name__)


class TemporalMemoryManager:
    """Orchestrates version-aware memory operations across SQLite and ChromaDB."""

    def __init__(
        self,
        sqlite_manager: Optional[SQLiteMemoryManager] = None,
        chroma_manager: Optional[ChromaMemoryManager] = None,
    ):
        self.sqlite = sqlite_manager or SQLiteMemoryManager()
        self.chroma = chroma_manager or ChromaMemoryManager()

    def ingest_fact(
        self,
        fact: ExtractedFact,
        session_id: str = "default_session",
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Process an extracted fact with version-aware logic:
        1. Check for existing active memory for (subject, attribute).
        2. If duplicate -> increment access count, do not create new version.
        3. If update -> atomically supersede old record and insert new record.
        4. If new -> insert as current record.
        Synchronizes both SQLite and ChromaDB.
        """
        effective_time = fact.valid_from or reference_time or get_current_utc()

        # Step 1: Look for existing current record
        existing = self.sqlite.get_current_fact(
            subject=fact.subject,
            attribute=fact.attribute,
            session_id=session_id,
        )

        if existing is None:
            # Case A: Brand new fact
            new_record = MemoryRecord(
                session_id=session_id,
                subject=fact.subject,
                attribute=fact.attribute,
                value=fact.value,
                memory_type=fact.memory_type,
                created_at=get_current_utc(),
                valid_from=effective_time,
                valid_until=None,
                confidence_score=fact.confidence_score,
                importance_score=fact.importance_score,
                status=MemoryStatus.CURRENT,
                supersedes_id=None,
                access_count=1,
                last_accessed=get_current_utc(),
                source_provenance=fact.source_provenance,
                raw_context=fact.raw_snippet,
            )
            self.sqlite.insert_memory(new_record)
            self.chroma.add_memory(new_record)

            logger.info(f"Created new memory: {new_record.subject}.{new_record.attribute} = '{new_record.value}'")
            return {
                "action": "created",
                "memory": new_record,
                "previous_memory": None,
            }

        # Step 2: Determine if this is a duplicate or an update
        if self._is_duplicate(existing.value, fact.value):
            # Case B: Duplicate fact
            self.sqlite.increment_access(existing.memory_id)
            logger.info(f"Duplicate fact detected: {existing.subject}.{existing.attribute}. Bumping access count.")
            return {
                "action": "duplicate_retained",
                "memory": existing,
                "previous_memory": None,
            }

        # Case C: Genuine update / state change
        new_record = MemoryRecord(
            session_id=session_id,
            subject=fact.subject,
            attribute=fact.attribute,
            value=fact.value,
            memory_type=fact.memory_type,
            created_at=get_current_utc(),
            valid_from=effective_time,
            valid_until=None,
            confidence_score=fact.confidence_score,
            importance_score=fact.importance_score,
            status=MemoryStatus.CURRENT,
            supersedes_id=existing.memory_id,
            access_count=1,
            last_accessed=get_current_utc(),
            source_provenance=fact.source_provenance,
            raw_context=fact.raw_snippet,
        )

        # Atomic transaction in SQLite
        self.sqlite.supersede_fact(
            old_memory_id=existing.memory_id,
            new_record=new_record,
            valid_until=effective_time,
        )

        # Update ChromaDB: update old record metadata, insert new record vector
        self.chroma.update_memory_status(
            memory_id=existing.memory_id,
            status=MemoryStatus.SUPERSEDED,
            valid_until_epoch=effective_time.timestamp(),
        )
        self.chroma.add_memory(new_record)

        logger.info(
            f"Updated memory: {new_record.subject}.{new_record.attribute} "
            f"from '{existing.value}' to '{new_record.value}' (superseded {existing.memory_id})"
        )
        return {
            "action": "superseded",
            "memory": new_record,
            "previous_memory": existing,
        }

    def ingest_facts(
        self,
        facts: List[ExtractedFact],
        session_id: str = "default_session",
        reference_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Process a collection of extracted facts sequentially."""
        results = []
        for fact in facts:
            res = self.ingest_fact(fact, session_id=session_id, reference_time=reference_time)
            results.append(res)
        return results

    def _is_duplicate(self, val_a: str, val_b: str) -> bool:
        """Check if two values represent identical factual assertions."""
        a = val_a.strip().lower()
        b = val_b.strip().lower()
        if a == b:
            return True
        # Strip common formatting like commas or currency symbols
        clean_a = a.replace(",", "").replace("$", "").replace("usd", "").strip()
        clean_b = b.replace(",", "").replace("$", "").replace("usd", "").strip()
        return clean_a == clean_b

    def get_current(
        self, subject: str, attribute: str, session_id: Optional[str] = None
    ) -> Optional[MemoryRecord]:
        """Fetch current active version."""
        return self.sqlite.get_current_fact(subject, attribute, session_id=session_id)

    def get_history(
        self, subject: str, attribute: str, session_id: Optional[str] = None
    ) -> List[MemoryRecord]:
        """Fetch full chronological history of a fact."""
        return self.sqlite.get_fact_history(subject, attribute, session_id=session_id)

    def get_lineage(self, memory_id: str) -> List[MemoryRecord]:
        """Fetch version lineage connected to memory_id."""
        return self.sqlite.get_lineage(memory_id)

    def mark_disputed(self, memory_id: str) -> bool:
        """Mark a memory as disputed in both SQLite and ChromaDB."""
        ok = self.sqlite.update_memory_status(memory_id, MemoryStatus.DISPUTED)
        if ok:
            self.chroma.update_memory_status(memory_id, MemoryStatus.DISPUTED)
        return ok

    def resolve_dispute(self, memory_id: str, new_status: MemoryStatus) -> bool:
        """Resolve a disputed memory to current or superseded."""
        ok = self.sqlite.update_memory_status(memory_id, new_status)
        if ok:
            self.chroma.update_memory_status(memory_id, new_status)
        return ok
