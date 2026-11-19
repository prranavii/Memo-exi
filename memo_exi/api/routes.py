"""
FastAPI REST API Routes for MEMO-EXI.
Implements all 11 required endpoints from Section 4 of the specification.
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from memo_exi.models.memory_models import (
    ExtractedFact,
    ExtractionResult,
    MemoryCreate,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    MemoryUpdate,
    QueryIntent,
    RetrievalCandidate,
    RetrievalResult,
)
from memo_exi.service import ConversationResponse, MEMOEXIEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# Service singleton
_engine: Optional[MEMOEXIEngine] = None


def get_engine() -> MEMOEXIEngine:
    global _engine
    if _engine is None:
        _engine = MEMOEXIEngine()
    return _engine


# -------------------------------------------------------------------------
# Request / Response Schemas
# -------------------------------------------------------------------------

class ConversationRequest(BaseModel):
    session_id: str = "default_session"
    message: str = Field(..., min_length=1)
    reference_time: Optional[datetime] = None
    token_budget: Optional[int] = None


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1)
    session_id: str = "default_session"
    reference_time: Optional[datetime] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    top_k: Optional[int] = 5
    token_budget: Optional[int] = 1024
    intent_override: Optional[QueryIntent] = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str = "default_session"
    token_budget: Optional[int] = 1024


class EvaluationRunRequest(BaseModel):
    scenario_ids: Optional[List[int]] = None
    baselines: Optional[List[str]] = None
    ablations: Optional[List[str]] = None


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------

@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_200_OK,
    summary="Process conversation input end-to-end",
)
def process_conversation(payload: ConversationRequest):
    """
    Accepts natural-language dialogue, extracts facts, checks contradictions,
    persists versioned memories, retrieves relevant context, and synthesizes grounded response.
    """
    try:
        engine = get_engine()
        return engine.process_conversation(
            session_id=payload.session_id,
            message=payload.message,
            reference_time=payload.reference_time,
            token_budget=payload.token_budget,
        )
    except Exception as e:
        logger.error(f"Error processing conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/memories/extract",
    response_model=ExtractionResult,
    status_code=status.HTTP_200_OK,
    summary="Extract structured memories from text without storing",
)
def extract_memories(payload: ExtractRequest):
    """Extract factual, preference, constraint, or episodic facts from natural text."""
    try:
        engine = get_engine()
        return engine.extractor.extract_facts(
            text=payload.text,
            session_id=payload.session_id,
            reference_time=payload.reference_time,
        )
    except Exception as e:
        logger.error(f"Error extracting facts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/memories",
    response_model=MemoryRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Store a new structured memory",
)
def store_memory(payload: MemoryCreate):
    """Explicitly insert a structured memory record into SQLite and ChromaDB."""
    try:
        engine = get_engine()
        fact = payload
        record = MemoryRecord(
            session_id=fact.session_id,
            subject=fact.subject,
            attribute=fact.attribute,
            value=fact.value,
            memory_type=fact.memory_type,
            valid_from=fact.valid_from or datetime.now(),
            confidence_score=fact.confidence_score,
            importance_score=fact.importance_score,
            status=MemoryStatus.CURRENT,
            source_provenance=fact.source_provenance,
            raw_context=fact.raw_context,
        )
        engine.sqlite.insert_memory(record)
        engine.chroma.add_memory(record)
        return record
    except Exception as e:
        logger.error(f"Error storing memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/memories",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Retrieve stored memories with filters",
)
def get_memories(
    session_id: Optional[str] = Query(None),
    status_filter: Optional[MemoryStatus] = Query(None, alias="status"),
    memory_type: Optional[MemoryType] = Query(None),
    subject: Optional[str] = Query(None),
    attribute: Optional[str] = Query(None),
    as_of_time: Optional[datetime] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Query memories with temporal point-in-time, session, and status filters."""
    try:
        engine = get_engine()
        memories = engine.sqlite.query_memories(
            session_id=session_id,
            status=status_filter,
            memory_type=memory_type,
            subject=subject,
            attribute=attribute,
            as_of_time=as_of_time,
            limit=limit,
            offset=offset,
        )
        return {"memories": memories, "count": len(memories), "limit": limit, "offset": offset}
    except Exception as e:
        logger.error(f"Error querying memories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/memories/history",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Retrieve historical versions of a fact",
)
def get_fact_history(
    subject: str = Query(..., min_length=1),
    attribute: str = Query(..., min_length=1),
    session_id: Optional[str] = Query(None),
):
    """Retrieve full chronological evolution of a given fact."""
    try:
        engine = get_engine()
        history = engine.temporal_mgr.get_history(
            subject=subject, attribute=attribute, session_id=session_id
        )
        return {"subject": subject, "attribute": attribute, "history": history, "total_versions": len(history)}
    except Exception as e:
        logger.error(f"Error getting fact history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/memories/{memory_id}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Retrieve individual memory and its lineage",
)
def get_memory_by_id(memory_id: str):
    """Retrieve a specific memory by ID along with its forward/backward version lineage."""
    try:
        engine = get_engine()
        memory = engine.sqlite.get_memory_by_id(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found.")
        lineage = engine.temporal_mgr.get_lineage(memory_id)
        return {"memory": memory, "lineage": lineage}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching memory {memory_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/memories/update",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Update an existing fact using version-aware logic",
)
def update_memory(payload: MemoryUpdate):
    """Version-aware update: marks old record superseded and creates new active version."""
    try:
        engine = get_engine()
        fact = ExtractedFact(
            subject=payload.subject,
            attribute=payload.attribute,
            value=payload.new_value,
            valid_from=payload.effective_time,
            confidence_score=payload.confidence_score,
            importance_score=payload.importance_score,
            source_provenance=payload.source_provenance,
            raw_snippet=payload.raw_context,
        )
        res = engine.temporal_mgr.ingest_fact(
            fact=fact,
            session_id=payload.session_id,
            reference_time=payload.effective_time,
        )
        return res
    except Exception as e:
        logger.error(f"Error updating memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/retrieval/search",
    response_model=RetrievalResult,
    status_code=status.HTTP_200_OK,
    summary="Perform semantic and temporal memory retrieval",
)
def search_retrieval(payload: SearchRequest):
    """Perform hybrid vector search, temporal routing, and context-budget optimization."""
    try:
        engine = get_engine()
        if payload.intent_override:
            from memo_exi.modules.classification.classifier import QueryClassificationResult
            classification = QueryClassificationResult(
                intent=payload.intent_override,
                raw_query=payload.query,
            )
        else:
            classification = engine.classifier.classify_query(payload.query)

        candidates = engine.retriever.retrieve(
            query=payload.query,
            classification=classification,
            session_id=payload.session_id,
            top_k=payload.top_k,
        )

        lineage_chains = {}
        if classification.intent == QueryIntent.TEMPORAL:
            lineage_chains = engine.retriever.retrieve_lineage_chains(candidates)

        return engine.budget_mgr.optimize_selection(
            query=payload.query,
            intent=classification.intent,
            candidates=candidates,
            token_budget=payload.token_budget,
            lineage_chains=lineage_chains,
        )
    except Exception as e:
        logger.error(f"Error in retrieval search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Generate responses using retrieved memories",
)
def chat_endpoint(payload: ChatRequest):
    """Generate grounded answer strictly based on retrieved temporal memory records."""
    try:
        engine = get_engine()
        conv_res = engine.process_conversation(
            session_id=payload.session_id,
            message=payload.query,
            token_budget=payload.token_budget,
        )
        return {
            "answer": conv_res.reply,
            "query_intent": conv_res.query_intent.value,
            "retrieved_memories_count": len(conv_res.retrieved_memories),
            "token_usage": conv_res.token_usage,
            "budget_compliant": conv_res.budget_compliant,
            "controller_decision": conv_res.controller_decision.model_dump(),
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/analytics",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Return memory statistics and retrieval analytics",
)
def get_analytics(session_id: Optional[str] = Query(None)):
    """Return system health, memory distribution by type and status, and vector counts."""
    try:
        engine = get_engine()
        db_stats = engine.sqlite.get_statistics(session_id=session_id)
        vector_count = engine.chroma.count()
        return {
            "session_id": session_id or "all_sessions",
            "sqlite_stats": db_stats,
            "chroma_vector_count": vector_count,
            "status": "healthy",
        }
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/evaluation/run",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Run benchmark scenarios and calculate evaluation metrics",
)
def run_evaluation(payload: EvaluationRunRequest):
    """Trigger benchmark scenarios and compute comparative metrics across baselines and ablations."""
    try:
        from memo_exi.evaluation.evaluator import BenchmarkRunner
        runner = BenchmarkRunner()
        results = runner.run_evaluation(
            scenario_ids=payload.scenario_ids,
            baselines=payload.baselines,
            ablations=payload.ablations,
        )
        return results
    except Exception as e:
        logger.error(f"Error running evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
