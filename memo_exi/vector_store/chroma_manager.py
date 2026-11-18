"""
ChromaDB Vector Store Manager.
Maintains dense vector embeddings linked to SQLite memory IDs with rich metadata filtering.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.utils import embedding_functions
from memo_exi.config import settings
from memo_exi.models.memory_models import MemoryRecord, MemoryStatus

logger = logging.getLogger(__name__)

# Far future epoch timestamp for facts that have no valid_until (currently active)
FAR_FUTURE_EPOCH = 32503680000.0  # Year 3000-01-01


class ChromaMemoryManager:
    """Manages dense semantic vector storage and similarity search using ChromaDB."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embedding_model_name = embedding_model_name or settings.embedding_model_name
        self._ensure_persist_dir()
        self._init_chroma()

    def _ensure_persist_dir(self) -> None:
        """Ensure the chroma persistence directory exists."""
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

    def _init_chroma(self) -> None:
        """Initialize ChromaDB client and collection with sentence transformer embedding function."""
        try:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Initialized ChromaDB collection '{self.collection_name}' at {self.persist_dir} "
                f"using model '{self.embedding_model_name}'"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            raise

    def add_memory(self, record: MemoryRecord) -> None:
        """Add or update a memory record vector and metadata in ChromaDB."""
        doc_text = record.to_embedding_text()
        valid_from_epoch = record.valid_from.timestamp()
        valid_until_epoch = (
            record.valid_until.timestamp() if record.valid_until else FAR_FUTURE_EPOCH
        )

        metadata = {
            "memory_id": record.memory_id,
            "session_id": record.session_id,
            "subject": record.subject,
            "attribute": record.attribute,
            "memory_type": record.memory_type.value,
            "status": record.status.value,
            "valid_from_epoch": float(valid_from_epoch),
            "valid_until_epoch": float(valid_until_epoch),
            "importance_score": float(record.importance_score),
            "confidence_score": float(record.confidence_score),
        }

        self.collection.upsert(
            ids=[record.memory_id],
            documents=[doc_text],
            metadatas=[metadata],
        )

    def update_memory_status(
        self,
        memory_id: str,
        status: MemoryStatus,
        valid_until_epoch: Optional[float] = None,
    ) -> None:
        """Update metadata for an existing memory in ChromaDB without recomputing embeddings."""
        try:
            existing = self.collection.get(ids=[memory_id], include=["metadatas"])
            if existing and existing["ids"] and existing["metadatas"]:
                meta = existing["metadatas"][0]
                meta["status"] = status.value
                if valid_until_epoch is not None:
                    meta["valid_until_epoch"] = float(valid_until_epoch)
                self.collection.update(ids=[memory_id], metadatas=[meta])
        except Exception as e:
            logger.warning(f"Could not update status for memory {memory_id} in ChromaDB: {e}")

    def query_similar(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        Returns a list of dicts: {'memory_id': str, 'similarity': float, 'distance': float, 'metadata': dict}
        """
        if self.collection.count() == 0:
            return []

        # Chroma throws if n_results > count
        n_results = min(top_k, self.collection.count())
        if n_results <= 0:
            return []

        query_args: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n_results,
            "include": ["metadatas", "distances", "documents"],
        }
        if where:
            query_args["where"] = where

        try:
            results = self.collection.query(**query_args)
        except Exception as e:
            logger.warning(f"Vector search failed with where={where}: {e}. Retrying without filter.")
            query_args.pop("where", None)
            results = self.collection.query(**query_args)

        formatted_results = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            ids = results["ids"][0]
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)

            for mem_id, dist, meta in zip(ids, distances, metadatas):
                # Cosine distance: similarity = 1 - distance (clamp to [0, 1])
                similarity = max(0.0, min(1.0, 1.0 - dist))
                formatted_results.append({
                    "memory_id": mem_id,
                    "similarity": similarity,
                    "distance": dist,
                    "metadata": meta,
                })

        return formatted_results

    def delete_memory(self, memory_id: str) -> None:
        """Delete a vector entry by memory_id."""
        try:
            self.collection.delete(ids=[memory_id])
        except Exception as e:
            logger.warning(f"Failed to delete memory {memory_id} from ChromaDB: {e}")

    def count(self) -> int:
        """Return total vector count in collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Reset or clear the vector collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.warning(f"Failed to reset ChromaDB collection: {e}")
