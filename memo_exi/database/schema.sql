-- MEMO-EXI Temporal Memory SQLite Schema
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

-- Optimization Indexes
CREATE INDEX IF NOT EXISTS idx_memories_subject_attr ON memories(subject, attribute);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_temporal ON memories(valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_supersedes ON memories(supersedes_id);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
