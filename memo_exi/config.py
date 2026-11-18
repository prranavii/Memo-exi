"""
Configuration management for MEMO-EXI using Pydantic Settings.
Loads configuration from environment variables and .env file.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Groq LLM API Settings
    groq_api_key: str = "mock_key_or_placeholder"
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Storage Paths
    sqlite_db_path: str = "./data/memo_exi.db"
    chroma_persist_dir: str = "./data/chroma_storage"
    chroma_collection_name: str = "memo_exi_vectors"

    # Embedding Model Settings
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Context Budget & Retrieval Defaults
    default_token_budget: int = 1024
    default_top_k: int = 5
    max_top_k: int = 15

    # Memory Utility Scoring Weights
    weight_relevance: float = 0.45
    weight_temporal: float = 0.25
    weight_importance: float = 0.20
    weight_redundancy: float = 0.10

    # Ambiguity and Conflict Thresholds
    conflict_similarity_threshold: float = 0.82
    confidence_delta_threshold: float = 0.15

    # Server Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        """Ensure necessary storage directories exist."""
        sqlite_dir = Path(self.sqlite_db_path).parent
        sqlite_dir.mkdir(parents=True, exist_ok=True)
        chroma_dir = Path(self.chroma_persist_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
