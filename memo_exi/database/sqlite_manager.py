"""
SQLite Temporal Memory Manager.
Manages transactional storage, versioning, temporal queries, and lineage tracking.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
from memo_exi.config import settings
from memo_exi.models.memory_models import (
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    get_current_utc,
)

logger = logging.getLogger(__name__)


class SQLiteMemoryManager:
    """ACID-compliant temporal ledger for structured agent memories."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.sqlite_db_path
        self._ensure_db_dir()
        self.init_db()

    def _ensure_db_dir(self) -> None:
        """Ensure the parent directory of the database exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection configured with foreign keys and Row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initialize the database schema and indexes."""
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()
        else:
            schema_sql = """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                attribute TEXT NOT NULL,
                value TEXT NOT NULL,
                memory_type TEXT NOT NULL CHECK(memory_type IN ('preference', 'factual', 'constraint', 'episodic')),
                created_at TIMESTAMP NOT NULL,
                valid_from TIMESTAMP NOT NULL,
                valid_until TIMESTAMP,
                confidence_score REAL NOT NULL CHECK(confidence_score >= 0.0 AND confidence_score <= 1.0),
                importance_score REAL NOT NULL CHECK(importance_score >= 0.0 AND importance_score <= 1.0),
                status TEXT NOT NULL CHECK(status IN ('current', 'superseded', 'disputed')),
                supersedes_id TEXT REFERENCES memories(memory_id),
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TIMESTAMP,
                source_provenance TEXT,
                raw_context TEXT,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_subject_attr ON memories(subject, attribute);
            CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
            CREATE INDEX IF NOT EXISTS idx_memories_temporal ON memories(valid_from, valid_until);
            CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
            CREATE INDEX IF NOT EXISTS idx_memories_supersedes ON memories(supersedes_id);
            CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
            """

        with self.get_connection() as conn:
            conn.executescript(schema_sql)
        logger.info(f"Initialized SQLite database schema at {self.db_path}")

    def insert_memory(self, record: MemoryRecord) -> MemoryRecord:
        """Insert a new memory record."""
        data = record.to_dict()
        sql = """
        INSERT INTO memories (
            memory_id, session_id, subject, attribute, value, memory_type,
            created_at, valid_from, valid_until, confidence_score, importance_score,
            status, supersedes_id, access_count, last_accessed, source_provenance,
            raw_context, metadata_json
        ) VALUES (
            :memory_id, :session_id, :subject, :attribute, :value, :memory_type,
            :created_at, :valid_from, :valid_until, :confidence_score, :importance_score,
            :status, :supersedes_id, :access_count, :last_accessed, :source_provenance,
            :raw_context, :metadata_json
        )
        """
        with self.get_connection() as conn:
            conn.execute(sql, data)
        return record

    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        """Fetch a single memory record by memory_id."""
        sql = "SELECT * FROM memories WHERE memory_id = ?"
        with self.get_connection() as conn:
            cursor = conn.execute(sql, (memory_id,))
            row = cursor.fetchone()
            if row:
                return MemoryRecord.from_sqlite_row(dict(row))
        return None

    def get_current_fact(
        self, subject: str, attribute: str, session_id: Optional[str] = None
    ) -> Optional[MemoryRecord]:
        """Fetch the currently active ('current') version of a fact."""
        params: List[Any] = [subject.strip(), attribute.strip().lower(), MemoryStatus.CURRENT.value]
        sql = """
        SELECT * FROM memories 
        WHERE LOWER(subject) = LOWER(?) AND LOWER(attribute) = LOWER(?) AND status = ?
        """
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        
        sql += " ORDER BY valid_from DESC LIMIT 1"

        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            row = cursor.fetchone()
            if row:
                return MemoryRecord.from_sqlite_row(dict(row))
        return None

    def supersede_fact(
        self, old_memory_id: str, new_record: MemoryRecord, valid_until: Optional[datetime] = None
    ) -> MemoryRecord:
        """
        Atomically supersede an existing memory record with a new version.
        Marks previous record as 'superseded' with valid_until timestamp.
        Inserts new record with status='current' and supersedes_id=old_memory_id.
        """
        effective_until = valid_until or new_record.valid_from or get_current_utc()
        new_record.supersedes_id = old_memory_id
        new_record.status = MemoryStatus.CURRENT

        with self.get_connection() as conn:
            # 1. Supersede previous record
            update_sql = """
            UPDATE memories 
            SET status = ?, valid_until = ? 
            WHERE memory_id = ?
            """
            conn.execute(
                update_sql,
                (MemoryStatus.SUPERSEDED.value, effective_until.isoformat(), old_memory_id),
            )

            # 2. Insert new record
            data = new_record.to_dict()
            insert_sql = """
            INSERT INTO memories (
                memory_id, session_id, subject, attribute, value, memory_type,
                created_at, valid_from, valid_until, confidence_score, importance_score,
                status, supersedes_id, access_count, last_accessed, source_provenance,
                raw_context, metadata_json
            ) VALUES (
                :memory_id, :session_id, :subject, :attribute, :value, :memory_type,
                :created_at, :valid_from, :valid_until, :confidence_score, :importance_score,
                :status, :supersedes_id, :access_count, :last_accessed, :source_provenance,
                :raw_context, :metadata_json
            )
            """
            conn.execute(insert_sql, data)

        logger.info(
            f"Fact superseded: memory_id {old_memory_id} superseded by {new_record.memory_id} "
            f"for ({new_record.subject}, {new_record.attribute})"
        )
        return new_record

    def update_memory_status(self, memory_id: str, new_status: MemoryStatus) -> bool:
        """Update the status of a memory (e.g., mark as 'disputed' or 'current')."""
        sql = "UPDATE memories SET status = ? WHERE memory_id = ?"
        with self.get_connection() as conn:
            cursor = conn.execute(sql, (new_status.value, memory_id))
            return cursor.rowcount > 0

    def query_memories(
        self,
        session_id: Optional[str] = None,
        status: Optional[MemoryStatus] = None,
        memory_type: Optional[MemoryType] = None,
        subject: Optional[str] = None,
        attribute: Optional[str] = None,
        as_of_time: Optional[datetime] = None,
        valid_from_start: Optional[datetime] = None,
        valid_from_end: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """
        Query memories with flexible temporal and relational filters.
        Supports point-in-time reconstruction (as_of_time).
        """
        clauses = []
        params: List[Any] = []

        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if status:
            clauses.append("status = ?")
            params.append(status.value)
        if memory_type:
            clauses.append("memory_type = ?")
            params.append(memory_type.value)
        if subject:
            clauses.append("LOWER(subject) = LOWER(?)")
            params.append(subject.strip())
        if attribute:
            clauses.append("LOWER(attribute) = LOWER(?)")
            params.append(attribute.strip())

        # Point-in-time temporal logic: valid_from <= as_of AND (valid_until IS NULL OR valid_until > as_of)
        if as_of_time:
            as_of_iso = as_of_time.isoformat()
            clauses.append("valid_from <= ? AND (valid_until IS NULL OR valid_until > ?)")
            params.extend([as_of_iso, as_of_iso])
        else:
            if valid_from_start:
                clauses.append("valid_from >= ?")
                params.append(valid_from_start.isoformat())
            if valid_from_end:
                clauses.append("valid_from <= ?")
                params.append(valid_from_end.isoformat())

        where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM memories {where_str} ORDER BY valid_from DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return [MemoryRecord.from_sqlite_row(dict(row)) for row in cursor.fetchall()]

    def get_fact_history(
        self, subject: str, attribute: str, session_id: Optional[str] = None
    ) -> List[MemoryRecord]:
        """Retrieve complete chronological version history of a given fact."""
        params: List[Any] = [subject.strip(), attribute.strip().lower()]
        sql = """
        SELECT * FROM memories 
        WHERE LOWER(subject) = LOWER(?) AND LOWER(attribute) = LOWER(?)
        """
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        
        sql += " ORDER BY valid_from ASC"

        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return [MemoryRecord.from_sqlite_row(dict(row)) for row in cursor.fetchall()]

    def get_lineage(self, memory_id: str) -> List[MemoryRecord]:
        """
        Retrieve the full version lineage (ancestors and descendants) for a memory_id.
        Traces back through supersedes_id and forward through memories superseding this one.
        """
        start = self.get_memory_by_id(memory_id)
        if not start:
            return []
        
        # Traverse backward to find root
        ancestors = []
        curr = start
        visited = {curr.memory_id}
        while curr.supersedes_id and curr.supersedes_id not in visited:
            parent = self.get_memory_by_id(curr.supersedes_id)
            if not parent:
                break
            ancestors.append(parent)
            visited.add(parent.memory_id)
            curr = parent
        
        ancestors.reverse()  # Oldest first
        
        # Traverse forward to find all descendants
        descendants = []
        curr_id = start.memory_id
        while True:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM memories WHERE supersedes_id = ? LIMIT 1", (curr_id,)
                )
                child_row = cursor.fetchone()
                if not child_row:
                    break
                child = MemoryRecord.from_sqlite_row(dict(child_row))
                if child.memory_id in visited:
                    break
                descendants.append(child)
                visited.add(child.memory_id)
                curr_id = child.memory_id

        return ancestors + [start] + descendants

    def increment_access(self, memory_id: str) -> None:
        """Bump access count and set last_accessed timestamp."""
        now_iso = get_current_utc().isoformat()
        sql = """
        UPDATE memories 
        SET access_count = access_count + 1, last_accessed = ? 
        WHERE memory_id = ?
        """
        with self.get_connection() as conn:
            conn.execute(sql, (now_iso, memory_id))

    def get_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Return operational statistics about the memory ledger."""
        params: List[Any] = []
        session_filter = ""
        if session_id:
            session_filter = "WHERE session_id = ?"
            params.append(session_id)

        with self.get_connection() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM memories {session_filter}", params).fetchone()[0]
            current_count = conn.execute(
                f"SELECT COUNT(*) FROM memories {session_filter} {'AND' if session_id else 'WHERE'} status = 'current'",
                params
            ).fetchone()[0]
            superseded_count = conn.execute(
                f"SELECT COUNT(*) FROM memories {session_filter} {'AND' if session_id else 'WHERE'} status = 'superseded'",
                params
            ).fetchone()[0]
            disputed_count = conn.execute(
                f"SELECT COUNT(*) FROM memories {session_filter} {'AND' if session_id else 'WHERE'} status = 'disputed'",
                params
            ).fetchone()[0]
            unique_subjects = conn.execute(
                f"SELECT COUNT(DISTINCT subject) FROM memories {session_filter}", params
            ).fetchone()[0]

            # Distribution by type
            type_rows = conn.execute(
                f"SELECT memory_type, COUNT(*) as cnt FROM memories {session_filter} GROUP BY memory_type",
                params
            ).fetchall()
            type_distribution = {row["memory_type"]: row["cnt"] for row in type_rows}

        return {
            "total_memories": total,
            "current_memories": current_count,
            "superseded_memories": superseded_count,
            "disputed_memories": disputed_count,
            "unique_subjects": unique_subjects,
            "type_distribution": type_distribution,
        }

    def clear_database(self) -> None:
        """Utility to wipe database for testing."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM memories;")
