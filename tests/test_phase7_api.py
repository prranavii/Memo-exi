"""
Integration tests for Phase 7: FastAPI REST API endpoints and service orchestration.
"""

from fastapi.testclient import TestClient
import pytest
from memo_exi.api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "MEMO-EXI"
    assert data["status"] == "operational"


def test_conversations_endpoint():
    """Verify end-to-end conversation flow via REST API."""
    payload = {
        "session_id": "test_api_session",
        "message": "My current salary is $130,000 per year.",
        "token_budget": 1024,
    }
    response = client.post("/api/v1/conversations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert data["session_id"] == "test_api_session"
    assert len(data["extracted_memories"]) >= 1


def test_extract_memories_endpoint():
    """Verify fact extraction endpoint."""
    payload = {
        "text": "I prefer using Python and pytest for all backend testing.",
        "session_id": "test_api_session",
    }
    response = client.post("/api/v1/memories/extract", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "facts" in data
    assert len(data["facts"]) >= 1


import uuid

def test_crud_memories_endpoints():
    """Verify explicit memory creation, query, get by ID, and history."""
    test_session = f"test_crud_session_{uuid.uuid4()}"

    # 1. Create explicit memory
    create_payload = {
        "session_id": test_session,
        "subject": "User",
        "attribute": "preferred_editor",
        "value": "Neovim",
        "memory_type": "preference",
        "confidence_score": 0.95,
        "importance_score": 0.7,
    }
    create_resp = client.post("/api/v1/memories", json=create_payload)
    assert create_resp.status_code == 201
    created_mem = create_resp.json()
    mem_id = created_mem["memory_id"]
    assert mem_id is not None

    # 2. Query memories
    query_resp = client.get(f"/api/v1/memories?session_id={test_session}")
    assert query_resp.status_code == 200
    mem_list = query_resp.json()["memories"]
    assert any(m["memory_id"] == mem_id for m in mem_list)

    # 3. Get individual memory by ID with lineage
    get_resp = client.get(f"/api/v1/memories/{mem_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["memory"]["memory_id"] == mem_id
    assert len(get_resp.json()["lineage"]) >= 1

    # 4. Version update endpoint
    update_payload = {
        "session_id": test_session,
        "subject": "User",
        "attribute": "preferred_editor",
        "new_value": "Helix",
    }
    update_resp = client.post("/api/v1/memories/update", json=update_payload)
    assert update_resp.status_code == 200
    assert update_resp.json()["action"] == "superseded"

    # 5. History endpoint
    hist_resp = client.get(
        f"/api/v1/memories/history?subject=User&attribute=preferred_editor&session_id={test_session}"
    )
    assert hist_resp.status_code == 200
    history = hist_resp.json()["history"]
    assert len(history) == 2
    assert [h["value"] for h in history] == ["Neovim", "Helix"]


def test_retrieval_search_endpoint():
    """Verify semantic search endpoint."""
    payload = {
        "query": "What editor do I use?",
        "session_id": "test_crud_session",
        "top_k": 3,
        "token_budget": 512,
    }
    response = client.post("/api/v1/retrieval/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "selected_memories" in data
    assert data["budget_compliant"] is True


def test_chat_endpoint():
    """Verify chat endpoint."""
    payload = {
        "query": "What is my current salary?",
        "session_id": "test_api_session",
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "query_intent" in data


def test_analytics_endpoint():
    """Verify system analytics endpoint."""
    response = client.get("/api/v1/analytics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "sqlite_stats" in data
    assert "chroma_vector_count" in data
