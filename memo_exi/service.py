"""
MEMO-EXI Core Service Orchestrator.
Coordinates Module A (Extraction), Module B (Temporal Manager), Module C (Classifier),
Module D (Retriever), Module E (Conflict Resolver), Module F (Context Budget),
Module G (Adaptive Controller), and Module H (Response Generator).
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from memo_exi.config import settings
from memo_exi.database.sqlite_manager import SQLiteMemoryManager
from memo_exi.llm.base import LLMProvider
from memo_exi.llm.factory import get_llm_provider
from memo_exi.models.memory_models import (
    ExtractedFact,
    ExtractionResult,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    QueryIntent,
    RetrievalCandidate,
    RetrievalResult,
    get_current_utc,
)
from memo_exi.modules.budget.budget_manager import ContextBudgetManager
from memo_exi.modules.classification.classifier import QueryClassificationResult, TemporalQueryClassifier
from memo_exi.modules.conflict.resolver import ConflictResolver
from memo_exi.modules.controller.controller import AdaptiveMemoryController, ControllerDecision
from memo_exi.modules.generation.generator import GeneratedResponse, ResponseGenerator
from memo_exi.modules.ingestion.extractor import FactExtractor
from memo_exi.modules.retrieval.retriever import SemanticTemporalRetriever
from memo_exi.modules.temporal.manager import TemporalMemoryManager
from memo_exi.vector_store.chroma_manager import ChromaMemoryManager

logger = logging.getLogger(__name__)


class ConversationResponse(BaseModel):
    """End-to-end response for conversation turn."""
    reply: str
    session_id: str
    query_intent: QueryIntent
    extracted_memories: List[MemoryRecord] = Field(default_factory=list)
    retrieved_memories: List[MemoryRecord] = Field(default_factory=list)
    utility_scores: List[Dict[str, Any]] = Field(default_factory=list)
    token_usage: int = 0
    token_budget: int = 1024
    budget_compliant: bool = True
    clarification_required: bool = False
    clarification_question: Optional[str] = None
    controller_decision: ControllerDecision


class MEMOEXIEngine:
    """Master engine orchestrating adaptive temporal memory operations."""

    def __init__(
        self,
        sqlite_manager: Optional[SQLiteMemoryManager] = None,
        chroma_manager: Optional[ChromaMemoryManager] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.llm = llm_provider or get_llm_provider()
        self.sqlite = sqlite_manager or SQLiteMemoryManager()
        self.chroma = chroma_manager or ChromaMemoryManager()

        # Wire all modules
        self.extractor = FactExtractor(llm_provider=self.llm)
        self.temporal_mgr = TemporalMemoryManager(sqlite_manager=self.sqlite, chroma_manager=self.chroma)
        self.classifier = TemporalQueryClassifier(llm_provider=self.llm)
        self.retriever = SemanticTemporalRetriever(sqlite_manager=self.sqlite, chroma_manager=self.chroma)
        self.conflict_resolver = ConflictResolver(llm_provider=self.llm)
        self.budget_mgr = ContextBudgetManager()
        self.controller = AdaptiveMemoryController()
        self.generator = ResponseGenerator(llm_provider=self.llm)

    def process_conversation(
        self,
        session_id: str,
        message: str,
        reference_time: Optional[datetime] = None,
        token_budget: Optional[int] = None,
    ) -> ConversationResponse:
        """
        Execute end-to-end adaptive memory lifecycle for a conversation message.
        """
        ref_time = reference_time or get_current_utc()
        budget = token_budget or settings.default_token_budget
        actions_log = []

        # ----------------------------------------------------
        # Step 1: Fact Extraction (Module A)
        # ----------------------------------------------------
        extraction_res = self.extractor.extract_facts(
            text=message,
            session_id=session_id,
            reference_time=ref_time,
        )

        extracted_records: List[MemoryRecord] = []
        clarification_required = False
        clarification_q = None

        if extraction_res.facts:
            actions_log.append(f"Extracted {len(extraction_res.facts)} factual claim(s).")
            for fact in extraction_res.facts:
                # ------------------------------------------------
                # Step 2: Conflict Evaluation (Module E)
                # ------------------------------------------------
                existing = self.sqlite.get_current_fact(
                    subject=fact.subject,
                    attribute=fact.attribute,
                    session_id=session_id,
                )

                if existing:
                    conflict_eval = self.conflict_resolver.analyze_update(existing, fact)
                    if conflict_eval.is_contradiction:
                        # Mark existing as disputed
                        self.temporal_mgr.mark_disputed(existing.memory_id)
                        clarification_required = True
                        clarification_q = conflict_eval.clarification_question
                        actions_log.append(f"Contradiction detected for ({fact.subject}, {fact.attribute}): marked disputed.")
                        continue

                # ------------------------------------------------
                # Step 3: Version-Aware Temporal Ingest (Module B)
                # ------------------------------------------------
                ingest_res = self.temporal_mgr.ingest_fact(
                    fact=fact,
                    session_id=session_id,
                    reference_time=ref_time,
                )
                if ingest_res.get("memory"):
                    extracted_records.append(ingest_res["memory"])
                actions_log.append(f"Memory action: {ingest_res.get('action')}")
        else:
            if extraction_res.filtered_out_chitchat:
                actions_log.append("Filtered non-informational conversational content.")

        # If a contradiction requires immediate user clarification, return early
        if clarification_required:
            controller_decision = ControllerDecision(
                initial_policy=self.controller.formulate_policy(QueryIntent.AMBIGUOUS, self.sqlite.get_statistics(session_id)["total_memories"], budget),
                actions_taken=actions_log,
            )
            return ConversationResponse(
                reply=clarification_q or "A contradiction was detected in your statement. Could you clarify?",
                session_id=session_id,
                query_intent=QueryIntent.AMBIGUOUS,
                extracted_memories=extracted_records,
                retrieved_memories=[],
                utility_scores=[],
                token_usage=0,
                token_budget=budget,
                budget_compliant=True,
                clarification_required=True,
                clarification_question=clarification_q,
                controller_decision=controller_decision,
            )

        # ----------------------------------------------------
        # Step 4: Query Intent Classification (Module C)
        # ----------------------------------------------------
        classification = self.classifier.classify_query(query=message, reference_time=ref_time)
        actions_log.append(f"Classified query intent as: {classification.intent.value}")

        # ----------------------------------------------------
        # Step 5: Adaptive Policy Decision (Module G)
        # ----------------------------------------------------
        stats = self.sqlite.get_statistics(session_id=session_id)
        total_memories = stats["total_memories"]

        policy = self.controller.formulate_policy(
            intent=classification.intent,
            total_memories=total_memories,
            context_budget=budget,
        )
        actions_log.append(f"Controller formulated policy: top_k={policy.top_k}, strategy={policy.filter_strategy}")

        # ----------------------------------------------------
        # Step 6: Memory Retrieval (Module D)
        # ----------------------------------------------------
        candidates = self.retriever.retrieve(
            query=message,
            classification=classification,
            session_id=session_id,
            top_k=policy.top_k,
        )

        # Evaluate retrieval quality & reformulate if sparse / poor
        reformulated = self.controller.evaluate_retrieval_and_adapt(
            query=message,
            intent=classification.intent,
            candidates=candidates,
        )

        retry_done = False
        if reformulated and policy.allow_reformulation:
            actions_log.append(f"Adaptive retry with reformulated query: '{reformulated}'")
            retry_candidates = self.retriever.retrieve(
                query=reformulated,
                classification=classification,
                session_id=session_id,
                top_k=policy.top_k,
            )
            if retry_candidates:
                candidates = retry_candidates
                retry_done = True

        # Fetch lineage chains if temporal query
        lineage_chains = {}
        if classification.intent == QueryIntent.TEMPORAL:
            lineage_chains = self.retriever.retrieve_lineage_chains(candidates)
            actions_log.append(f"Retrieved lineage chains for {len(lineage_chains)} entities.")

        # ----------------------------------------------------
        # Step 7: Context-Budget Optimization (Module F)
        # ----------------------------------------------------
        budget_result = self.budget_mgr.optimize_selection(
            query=message,
            intent=classification.intent,
            candidates=candidates,
            token_budget=budget,
            lineage_chains=lineage_chains,
        )
        actions_log.append(
            f"Budget selection: packed {len(budget_result.selected_memories)} memories "
            f"({budget_result.total_tokens_used}/{budget} tokens)"
        )

        # ----------------------------------------------------
        # Step 8: Response Generation (Module H)
        # ----------------------------------------------------
        gen_response = self.generator.generate_response(
            query=message,
            retrieval_result=budget_result,
            reference_time=ref_time,
        )

        utility_scores_log = [
            {
                "memory_id": c.memory.memory_id,
                "fact": f"{c.memory.subject} {c.memory.attribute}: {c.memory.value}",
                "status": c.memory.status.value,
                "utility": round(c.utility_score, 4),
                "relevance": round(c.relevance_score, 4),
                "temporal": round(c.temporal_score, 4),
                "importance": round(c.importance_score, 4),
                "tokens": c.estimated_tokens,
                "selected": c.selected,
            }
            for c in budget_result.candidates
        ]

        controller_decision = ControllerDecision(
            initial_policy=policy,
            reformulated_query=reformulated,
            retry_executed=retry_done,
            final_candidate_count=len(budget_result.candidates),
            actions_taken=actions_log,
        )

        return ConversationResponse(
            reply=gen_response.answer,
            session_id=session_id,
            query_intent=classification.intent,
            extracted_memories=extracted_records,
            retrieved_memories=budget_result.selected_memories,
            utility_scores=utility_scores_log,
            token_usage=budget_result.total_tokens_used,
            token_budget=budget,
            budget_compliant=budget_result.budget_compliant,
            clarification_required=False,
            clarification_question=None,
            controller_decision=controller_decision,
        )
