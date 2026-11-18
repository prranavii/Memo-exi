"""
Unit and integration tests for Phase 1: SQLite schema, temporal operations, and ChromaDB vector store.
"""

from datetime import datetime, timedelta, timezone
import pytest
from memo_exi.models.memory_models import (
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    get_current_utc,
)


def test_sqlite_schema_initialization(sqlite_manager):
    """Verify that tables and indexes are created properly."""
    with sqlite_manager.get_connection() as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories';"
        )
        assert cursor.fetchone() is not None

        # Check indexes
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_memories%';"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        assert "idx_memories_subject_attr" in indexes
        assert "idx_memories_status" in indexes
        assert "idx_memories_temporal" in indexes


def test_sqlite_insert_and_retrieve(sqlite_manager):
    """Verify memory record insertion and retrieval by ID."""
    record = MemoryRecord(
        subject="User",
        attribute="salary",
        value="$90,000",
        memory_type=MemoryType.FACTUAL,
        confidence_score=0.95,
        importance_score=0.8,
    )
    inserted = sqlite_manager.insert_memory(record)
    assert inserted.memory_id == record.memory_id

    fetched = sqlite_manager.get_memory_by_id(record.memory_id)
    assert fetched is not None
    assert fetched.subject == "User"
    assert fetched.attribute == "salary"
    assert fetched.value == "$90,000"
    assert fetched.status == MemoryStatus.CURRENT
    assert fetched.valid_until is None


def test_sqlite_supersede_fact_transaction(sqlite_manager):
    """Verify atomic superseding: old is superseded with valid_until, new is current with supersedes_id."""
    t0 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    t1 = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)

    # Version 1: Salary = $80,000
    v1 = MemoryRecord(
        subject="User",
        attribute="salary",
        value="$80,000",
        valid_from=t0,
        status=MemoryStatus.CURRENT,
    )
    sqlite_manager.insert_memory(v1)

    # Verify v1 is current
    curr = sqlite_manager.get_current_fact("User", "salary")
    assert curr is not None
    assert curr.value == "$80,000"

    # Version 2: Salary = $95,000 at t1
    v2 = MemoryRecord(
        subject="User",
        attribute="salary",
        value="$95,000",
        valid_from=t1,
    )
    sqlite_manager.supersede_fact(old_memory_id=v1.memory_id, new_record=v2, valid_until=t1)

    # Verify v1 is now superseded
    v1_reloaded = sqlite_manager.get_memory_by_id(v1.memory_id)
    assert v1_reloaded.status == MemoryStatus.SUPERSEDED
    assert v1_reloaded.valid_until is not None
    assert v1_reloaded.valid_until == t1

    # Verify v2 is now current and linked
    v2_reloaded = sqlite_manager.get_memory_by_id(v2.memory_id)
    assert v2_reloaded.status == MemoryStatus.CURRENT
    assert v2_reloaded.supersedes_id == v1.memory_id

    # Verify get_current_fact returns v2
    curr_now = sqlite_manager.get_current_fact("User", "salary")
    assert curr_now.memory_id == v2.memory_id
    assert curr_now.value == "$95,000"


def test_point_in_time_query(sqlite_manager):
    """Verify point-in-time temporal reconstruction (as_of_time query)."""
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2024, 5, 1, tzinfo=timezone.utc)
    t2 = datetime(2024, 9, 1, tzinfo=timezone.utc)

    v1 = MemoryRecord(subject="User", attribute="role", value="Junior Dev", valid_from=t0)
    sqlite_manager.insert_memory(v1)

    v2 = MemoryRecord(subject="User", attribute="role", value="Mid-level Dev", valid_from=t1)
    sqlite_manager.supersede_fact(v1.memory_id, v2, valid_until=t1)

    v3 = MemoryRecord(subject="User", attribute="role", value="Senior Dev", valid_from=t2)
    sqlite_manager.supersede_fact(v2.memory_id, v3, valid_until=t2)

    # Query as of March 2024 (should be Junior Dev)
    as_of_march = datetime(2024, 3, 1, tzinfo=timezone.utc)
    res_march = sqlite_manager.query_memories(subject="User", attribute="role", as_of_time=as_of_march)
    assert len(res_march) == 1
    assert res_march[0].value == "Junior Dev"

    # Query as of July 2024 (should be Mid-level Dev)
    as_of_july = datetime(2024, 7, 1, tzinfo=timezone.utc)
    res_july = sqlite_manager.query_memories(subject="User", attribute="role", as_of_time=as_of_july)
    assert len(res_july) == 1
    assert res_july[0].value == "Mid-level Dev"

    # Query as of October 2024 (should be Senior Dev)
    as_of_oct = datetime(2024, 10, 1, tzinfo=timezone.utc)
    res_oct = sqlite_manager.query_memories(subject="User", attribute="role", as_of_time=as_of_oct)
    assert len(res_oct) == 1
    assert res_oct[0].value == "Senior Dev"


def test_fact_history_and_lineage(sqlite_manager):
    """Verify version history and forward/backward lineage traversal."""
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2024, 2, 1, tzinfo=timezone.utc)
    t2 = datetime(2024, 3, 1, tzinfo=timezone.utc)

    v1 = MemoryRecord(subject="Project", attribute="deadline", value="2024-06-01", valid_from=t0)
    sqlite_manager.insert_memory(v1)

    v2 = MemoryRecord(subject="Project", attribute="deadline", value="2024-07-15", valid_from=t1)
    sqlite_manager.supersede_fact(v1.memory_id, v2, valid_until=t1)

    v3 = MemoryRecord(subject="Project", attribute="deadline", value="2024-09-01", valid_from=t2)
    sqlite_manager.supersede_fact(v2.memory_id, v3, valid_until=t2)

    history = sqlite_manager.get_fact_history("Project", "deadline")
    assert len(history) == 3
    assert [h.value for h in history] == ["2024-06-01", "2024-07-15", "2024-09-01"]

    # Lineage from intermediate node v2
    lineage = sqlite_manager.get_lineage(v2.memory_id)
    assert len(lineage) == 3
    assert lineage[0].memory_id == v1.memory_id
    assert lineage[1].memory_id == v2.memory_id
    assert lineage[2].memory_id == v3.memory_id


def test_chroma_vector_operations(chroma_manager):
    """Verify ChromaDB vector insertion, similarity search, and status filtering."""
    r1 = MemoryRecord(
        subject="User",
        attribute="framework",
        value="FastAPI and PyTorch",
        memory_type=MemoryType.PREFERENCE,
        status=MemoryStatus.CURRENT,
    )
    r2 = MemoryRecord(
        subject="User",
        attribute="car",
        value="Tesla Model 3",
        memory_type=MemoryType.FACTUAL,
        status=MemoryStatus.CURRENT,
    )
    r3 = MemoryRecord(
        subject="User",
        attribute="framework",
        value="Django",
        memory_type=MemoryType.PREFERENCE,
        status=MemoryStatus.SUPERSEDED,
    )

    chroma_manager.add_memory(r1)
    chroma_manager.add_memory(r2)
    chroma_manager.add_memory(r3)

    assert chroma_manager.count() == 3

    # Query for programming frameworks
    matches = chroma_manager.query_similar("What Python framework do I prefer?", top_k=2)
    assert len(matches) > 0
    # Top match should be framework related (r1 or r3)
    top_id = matches[0]["memory_id"]
    assert top_id in [r1.memory_id, r3.memory_id]

    # Query with metadata filter for current status only
    current_matches = chroma_manager.query_similar(
        "Python web framework", top_k=5, where={"status": "current"}
    )
    assert all(m["metadata"]["status"] == "current" for m in current_matches)
    # Django (superseded) should not be present
    assert not any(m["memory_id"] == r3.memory_id for m in current_matches)


def test_sqlite_and_chroma_sync(sqlite_manager, chroma_manager):
    """Verify dual-engine synchronization between SQLite and ChromaDB."""
    r = MemoryRecord(
        subject="User",
        attribute="location",
        value="San Francisco, CA",
        status=MemoryStatus.CURRENT,
    )
    sqlite_manager.insert_memory(r)
    chroma_manager.add_memory(r)

    # Search in Chroma, fetch authoritative data from SQLite
    matches = chroma_manager.query_similar("Where do I reside?", top_k=1)
    assert len(matches) == 1
    found_id = matches[0]["memory_id"]

    authoritative = sqlite_manager.get_memory_by_id(found_id)
    assert authoritative is not None
    assert authoritative.value == "San Francisco, CA"
    assert authoritative.status == MemoryStatus.CURRENT
