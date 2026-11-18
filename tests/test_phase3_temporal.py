"""
Unit and integration tests for Phase 3: Temporal Memory Manager (Module B).
"""

from datetime import datetime, timezone
import pytest
from memo_exi.models.memory_models import ExtractedFact, MemoryStatus, MemoryType
from memo_exi.modules.temporal.manager import TemporalMemoryManager


@pytest.fixture
def temporal_manager(sqlite_manager, chroma_manager):
    """Instantiate TemporalMemoryManager wired to test storage fixtures."""
    return TemporalMemoryManager(sqlite_manager=sqlite_manager, chroma_manager=chroma_manager)


def test_ingest_new_fact(temporal_manager):
    """Verify ingestion of a new fact creates records in SQLite and ChromaDB."""
    fact = ExtractedFact(
        subject="User",
        attribute="city",
        value="Seattle",
        memory_type=MemoryType.FACTUAL,
        confidence_score=0.95,
        importance_score=0.7,
    )
    res = temporal_manager.ingest_fact(fact, session_id="s1")
    assert res["action"] == "created"
    assert res["memory"].status == MemoryStatus.CURRENT

    # Check SQLite
    current = temporal_manager.get_current("User", "city", session_id="s1")
    assert current is not None
    assert current.value == "Seattle"

    # Check ChromaDB
    similar = temporal_manager.chroma.query_similar("Where does the user live?")
    assert len(similar) > 0
    assert similar[0]["memory_id"] == current.memory_id


def test_ingest_duplicate_fact(temporal_manager):
    """Verify duplicate facts do not create redundant versions, but bump access count."""
    fact1 = ExtractedFact(
        subject="User",
        attribute="editor",
        value="VS Code",
        memory_type=MemoryType.PREFERENCE,
    )
    res1 = temporal_manager.ingest_fact(fact1, session_id="s1")
    assert res1["action"] == "created"
    first_id = res1["memory"].memory_id

    # Ingest same value with minor whitespace/casing
    fact2 = ExtractedFact(
        subject="User",
        attribute="editor",
        value="vs code",
        memory_type=MemoryType.PREFERENCE,
    )
    res2 = temporal_manager.ingest_fact(fact2, session_id="s1")
    assert res2["action"] == "duplicate_retained"
    assert res2["memory"].memory_id == first_id

    # Total versions in history should still be 1
    history = temporal_manager.get_history("User", "editor", session_id="s1")
    assert len(history) == 1
    assert history[0].access_count >= 1


def test_iterative_version_updates(temporal_manager):
    """Verify iterative state changes preserve full lineage without destructive overwrites."""
    t0 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    t1 = datetime(2024, 4, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2024, 8, 1, 10, 0, tzinfo=timezone.utc)

    # V1
    f1 = ExtractedFact(subject="User", attribute="salary", value="$80,000", valid_from=t0)
    temporal_manager.ingest_fact(f1, session_id="s1")

    # V2
    f2 = ExtractedFact(subject="User", attribute="salary", value="$95,000", valid_from=t1)
    res2 = temporal_manager.ingest_fact(f2, session_id="s1")
    assert res2["action"] == "superseded"

    # V3
    f3 = ExtractedFact(subject="User", attribute="salary", value="$115,000", valid_from=t2)
    res3 = temporal_manager.ingest_fact(f3, session_id="s1")
    assert res3["action"] == "superseded"

    # Verify history
    history = temporal_manager.get_history("User", "salary", session_id="s1")
    assert len(history) == 3
    assert [h.value for h in history] == ["$80,000", "$95,000", "$115,000"]
    assert [h.status for h in history] == [
        MemoryStatus.SUPERSEDED,
        MemoryStatus.SUPERSEDED,
        MemoryStatus.CURRENT,
    ]

    # Verify lineage links
    assert history[1].supersedes_id == history[0].memory_id
    assert history[2].supersedes_id == history[1].memory_id

    # Check active current
    active = temporal_manager.get_current("User", "salary", session_id="s1")
    assert active.value == "$115,000"


def test_dispute_workflow(temporal_manager):
    """Verify marking memories as disputed and subsequent resolution."""
    f = ExtractedFact(subject="Server", attribute="ip", value="192.168.1.10")
    res = temporal_manager.ingest_fact(f, session_id="s1")
    mem_id = res["memory"].memory_id

    # Mark as disputed
    ok = temporal_manager.mark_disputed(mem_id)
    assert ok is True

    record = temporal_manager.sqlite.get_memory_by_id(mem_id)
    assert record.status == MemoryStatus.DISPUTED

    # Resolve dispute back to current
    ok_res = temporal_manager.resolve_dispute(mem_id, MemoryStatus.CURRENT)
    assert ok_res is True
    record_resolved = temporal_manager.sqlite.get_memory_by_id(mem_id)
    assert record_resolved.status == MemoryStatus.CURRENT
