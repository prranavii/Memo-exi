"""
Pytest configuration and shared fixtures for MEMO-EXI test suite.
"""

import shutil
import tempfile
import pytest
from memo_exi.database.sqlite_manager import SQLiteMemoryManager
from memo_exi.vector_store.chroma_manager import ChromaMemoryManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for isolated test execution."""
    tmp = tempfile.mkdtemp(prefix="memo_exi_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sqlite_manager(temp_dir):
    """Instantiate a fresh SQLite manager in a temp directory."""
    db_path = f"{temp_dir}/test_memo_exi.db"
    manager = SQLiteMemoryManager(db_path=db_path)
    return manager


@pytest.fixture
def chroma_manager(temp_dir):
    """Instantiate a fresh ChromaDB manager in a temp directory."""
    chroma_dir = f"{temp_dir}/test_chroma"
    manager = ChromaMemoryManager(
        persist_dir=chroma_dir,
        collection_name="test_memo_vectors",
        embedding_model_name="all-MiniLM-L6-v2",
    )
    return manager
