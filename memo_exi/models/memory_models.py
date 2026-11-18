"""
Core Pydantic models for MEMO-EXI memory records, extraction, retrieval, and lineage.
"""

from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, field_validator


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    FACTUAL = "factual"
    CONSTRAINT = "constraint"
    EPISODIC = "episodic"


class MemoryStatus(str, Enum):
    CURRENT = "current"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"


class QueryIntent(str, Enum):
    CURRENT_STATE = "current_state"
    HISTORICAL = "historical"
    TEMPORAL = "temporal"
    GENERAL_SEMANTIC = "general_semantic"
    AMBIGUOUS = "ambiguous"


def get_current_utc() -> datetime:
    return datetime.now(timezone.utc)


class MemoryRecord(BaseModel):
    """Authoritative representation of a structured temporal memory record."""
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default="default_session")
    subject: str = Field(..., min_length=1, description="Entity or topic the memory describes")
    attribute: str = Field(..., min_length=1, description="Specific attribute, relationship, or property")
    value: str = Field(..., min_length=1, description="Value or assertion of the fact")
    memory_type: MemoryType = Field(default=MemoryType.FACTUAL)
    created_at: datetime = Field(default_factory=get_current_utc)
    valid_from: datetime = Field(default_factory=get_current_utc)
    valid_until: Optional[datetime] = None
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    status: MemoryStatus = Field(default=MemoryStatus.CURRENT)
    supersedes_id: Optional[str] = None
    access_count: int = Field(default=0, ge=0)
    last_accessed: Optional[datetime] = None
    source_provenance: Optional[str] = None
    raw_context: Optional[str] = None
    metadata_json: Optional[str] = None

    def to_embedding_text(self) -> str:
        """Textual representation formatted for dense vector retrieval."""
        return f"{self.subject} {self.attribute}: {self.value} (type: {self.memory_type.value})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for SQLite row binding."""
        return {
            "memory_id": self.memory_id,
            "session_id": self.session_id,
            "subject": self.subject.strip(),
            "attribute": self.attribute.strip().lower(),
            "value": self.value.strip(),
            "memory_type": self.memory_type.value,
            "created_at": self.created_at.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "confidence_score": float(self.confidence_score),
            "importance_score": float(self.importance_score),
            "status": self.status.value,
            "supersedes_id": self.supersedes_id,
            "access_count": int(self.access_count),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "source_provenance": self.source_provenance,
            "raw_context": self.raw_context,
            "metadata_json": self.metadata_json or json.dumps({}),
        }

    @classmethod
    def from_sqlite_row(cls, row: Dict[str, Any]) -> "MemoryRecord":
        """Reconstruct model from SQLite row mapping."""
        def parse_dt(val: Any) -> Optional[datetime]:
            if not val:
                return None
            if isinstance(val, datetime):
                return val
            try:
                return datetime.fromisoformat(val)
            except Exception:
                return None

        return cls(
            memory_id=row["memory_id"],
            session_id=row["session_id"],
            subject=row["subject"],
            attribute=row["attribute"],
            value=row["value"],
            memory_type=MemoryType(row["memory_type"]),
            created_at=parse_dt(row["created_at"]) or get_current_utc(),
            valid_from=parse_dt(row["valid_from"]) or get_current_utc(),
            valid_until=parse_dt(row["valid_until"]),
            confidence_score=float(row["confidence_score"]),
            importance_score=float(row["importance_score"]),
            status=MemoryStatus(row["status"]),
            supersedes_id=row["supersedes_id"],
            access_count=int(row.get("access_count", 0)),
            last_accessed=parse_dt(row.get("last_accessed")),
            source_provenance=row.get("source_provenance"),
            raw_context=row.get("raw_context"),
            metadata_json=row.get("metadata_json"),
        )


class MemoryCreate(BaseModel):
    """Payload to create an explicit memory record."""
    session_id: str = "default_session"
    subject: str = Field(..., min_length=1)
    attribute: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)
    memory_type: MemoryType = MemoryType.FACTUAL
    valid_from: Optional[datetime] = None
    confidence_score: float = Field(default=0.9, ge=0.0, le=1.0)
    importance_score: float = Field(default=0.6, ge=0.0, le=1.0)
    source_provenance: Optional[str] = "user_input"
    raw_context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryUpdate(BaseModel):
    """Payload to update an existing fact with version tracking."""
    session_id: str = "default_session"
    subject: str = Field(..., min_length=1)
    attribute: str = Field(..., min_length=1)
    new_value: str = Field(..., min_length=1)
    effective_time: Optional[datetime] = None
    confidence_score: float = Field(default=0.95, ge=0.0, le=1.0)
    importance_score: float = Field(default=0.7, ge=0.0, le=1.0)
    source_provenance: Optional[str] = "user_update"
    raw_context: Optional[str] = None


class ExtractedFact(BaseModel):
    """A structured factual claim extracted from natural language interaction."""
    subject: str = Field(..., description="Entity or agent described (e.g., 'User', 'Project X')")
    attribute: str = Field(..., description="Property or relationship (e.g., 'salary', 'deadline', 'favorite_editor')")
    value: str = Field(..., description="Factual value or statement")
    memory_type: MemoryType = Field(default=MemoryType.FACTUAL)
    confidence_score: float = Field(default=0.85, ge=0.0, le=1.0)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    valid_from: Optional[datetime] = None
    source_provenance: Optional[str] = None
    raw_snippet: Optional[str] = None


class ExtractionResult(BaseModel):
    """Wrapper for multiple facts extracted from a single interaction."""
    session_id: str
    facts: List[ExtractedFact] = Field(default_factory=list)
    raw_message: str
    filtered_out_chitchat: bool = False


class TemporalFilter(BaseModel):
    """Filter specification for point-in-time and interval queries."""
    as_of: Optional[datetime] = None
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None
    only_current: bool = True
    include_disputed: bool = False


class RetrievalCandidate(BaseModel):
    """A retrieved memory candidate evaluated for utility and budget packing."""
    memory: MemoryRecord
    relevance_score: float = 0.0
    temporal_score: float = 1.0
    importance_score: float = 0.5
    redundancy_penalty: float = 0.0
    utility_score: float = 0.0
    estimated_tokens: int = 0
    selected: bool = False


class RetrievalResult(BaseModel):
    """Final output from retrieval & context budget ranking."""
    query: str
    intent: QueryIntent
    candidates: List[RetrievalCandidate] = Field(default_factory=list)
    selected_memories: List[MemoryRecord] = Field(default_factory=list)
    total_tokens_used: int = 0
    token_budget: int = 1024
    budget_compliant: bool = True
    lineage_chains: Dict[str, List[MemoryRecord]] = Field(default_factory=dict)


class LineageChain(BaseModel):
    """Chain of memory versions representing the chronological evolution of a fact."""
    subject: str
    attribute: str
    versions: List[MemoryRecord] = Field(default_factory=list)
    current_version: Optional[MemoryRecord] = None
