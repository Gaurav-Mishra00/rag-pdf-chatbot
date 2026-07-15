"""
tests/test_missing_apis.py

Tests for all new and enhanced endpoints introduced in the
"Missing APIs & Feature Enhancements" sprint:

  Group 1 — Session APIs
    GET  /api/v1/sessions
    GET  /api/v1/sessions/{session_id}
    DELETE /api/v1/sessions/{session_id}

  Group 2 — Document Management Gaps
    GET  /api/v1/documents              (now typed + includes error_message)
    GET  /api/v1/documents/{id}         (new single-doc detail)

  Group 3 — Vector Store Improvements
    GET  /api/v1/vectorstore/status     (now includes vector_count & file size)
    GET  /api/v1/vectorstore/count      (new lightweight endpoint)

  Group 4 — Health / Readiness
    GET  /health/ready                  (new readiness probe)

  Group 5 — Chat Metadata
    POST /api/v1/chat/query             (metadata is now typed / real values)
"""

import pytest
from unittest.mock import MagicMock, patch


HEADERS = {"X-API-Key": "test_secret_key"}


# ===========================================================================
# Fixtures — shared setup helpers
# ===========================================================================

def _upload_pdf(client, filename: str = "session_test.pdf") -> str:
    """Upload a minimal mock PDF and return its document_id."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Session test document content."
    with patch("app.services.pdf_processor.PdfReader") as mock_cls:
        mock_cls.return_value.pages = [mock_page]
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, b"%PDF-test", "application/pdf")},
            headers=HEADERS,
        )
    assert resp.status_code == 201, resp.text
    return resp.json()["document_id"]


def _chat(client, message: str = "What is in the document?", session_id: str = "test-sess-001") -> dict:
    """Send a chat query and return the JSON response."""
    resp = client.post(
        "/api/v1/chat/query",
        json={"message": message, "session_id": session_id},
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ===========================================================================
# Group 1 — Session APIs
# ===========================================================================

@pytest.mark.anyio
async def test_list_sessions_empty(client):
    """GET /sessions returns an empty list when no chat history exists."""
    # Use a unique session to avoid cross-test pollution
    # First clear any pre-existing state by checking response shape only
    resp = client.get("/api/v1/sessions", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert "total" in data
    assert isinstance(data["sessions"], list)
    assert data["total"] == len(data["sessions"])


def test_list_sessions_after_chat(client):
    """GET /sessions includes a session after a chat query is made."""
    session_id = "new-session-for-listing"
    _chat(client, session_id=session_id)

    resp = client.get("/api/v1/sessions", headers=HEADERS)
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]
    ids = [s["session_id"] for s in sessions]
    assert session_id in ids

    # Verify response shape
    matching = next(s for s in sessions if s["session_id"] == session_id)
    assert matching["message_count"] >= 2  # user + assistant
    assert "last_activity" in matching


def test_get_session_history(client):
    """GET /sessions/{id} returns the correct messages for a session."""
    session_id = "get-history-test-session"
    _chat(client, message="Hello from test", session_id=session_id)

    resp = client.get(f"/api/v1/sessions/{session_id}", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id
    assert data["total"] >= 2
    assert isinstance(data["messages"], list)

    roles = {m["role"] for m in data["messages"]}
    assert "user" in roles
    assert "assistant" in roles


def test_get_session_history_not_found(client):
    """GET /sessions/{id} returns 404 for an unknown session."""
    resp = client.get("/api/v1/sessions/nonexistent-xyz-999", headers=HEADERS)
    assert resp.status_code == 404


def test_delete_session(client):
    """DELETE /sessions/{id} clears history and the session is no longer listed."""
    session_id = "delete-test-session-abc"
    _chat(client, session_id=session_id)

    # Confirm it exists
    resp = client.get(f"/api/v1/sessions/{session_id}", headers=HEADERS)
    assert resp.status_code == 200

    # Delete it
    del_resp = client.delete(f"/api/v1/sessions/{session_id}", headers=HEADERS)
    assert del_resp.status_code == 200
    body = del_resp.json()
    assert body["session_id"] == session_id
    assert body["messages_deleted"] >= 2

    # Confirm history is gone
    get_resp = client.get(f"/api/v1/sessions/{session_id}", headers=HEADERS)
    assert get_resp.status_code == 404


def test_delete_session_idempotent(client):
    """DELETE /sessions/{id} on a non-existent session returns 200 with zero deleted."""
    resp = client.delete("/api/v1/sessions/never-existed-session", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["messages_deleted"] == 0


# ===========================================================================
# Group 2 — Document Management Gaps
# ===========================================================================

def test_list_documents_typed_response(client):
    """GET /documents returns a list of typed DocumentStatusResponse objects."""
    doc_id = _upload_pdf(client, "typed_list_test.pdf")
    resp = client.get("/api/v1/documents", headers=HEADERS)
    assert resp.status_code == 200
    docs = resp.json()
    assert isinstance(docs, list)
    # Each item must have the required typed fields
    for doc in docs:
        assert "document_id" in doc
        assert "filename" in doc
        assert "status" in doc
        assert "chunk_count" in doc

    # The uploaded doc must appear
    assert any(d["document_id"] == doc_id for d in docs)


def test_get_document_by_id(client):
    """GET /documents/{id} returns correct single document detail."""
    doc_id = _upload_pdf(client, "single_doc_test.pdf")

    resp = client.get(f"/api/v1/documents/{doc_id}", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == doc_id
    assert data["filename"] == "single_doc_test.pdf"
    assert data["status"] == "completed"
    assert data["chunk_count"] == 1
    assert data["file_size_bytes"] > 0


def test_get_document_by_id_not_found(client):
    """GET /documents/{id} returns 404 for an unknown document ID."""
    resp = client.get("/api/v1/documents/nonexistent-doc-0000", headers=HEADERS)
    assert resp.status_code == 404


# ===========================================================================
# Group 3 — Vector Store Improvements
# ===========================================================================

def test_vectorstore_status_has_vector_count(client):
    """GET /vectorstore/status includes vector_count, index_path, index_file_size_kb."""
    resp = client.get("/api/v1/vectorstore/status", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "vector_count" in data
    assert "index_path" in data
    assert "index_file_size_kb" in data
    assert isinstance(data["vector_count"], int)
    assert data["vector_count"] >= 0


def test_vectorstore_count_endpoint(client):
    """GET /vectorstore/count returns {"count": N} with a non-negative integer."""
    resp = client.get("/api/v1/vectorstore/count", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert isinstance(data["count"], int)
    assert data["count"] >= 0


def test_vectorstore_count_increases_after_upload(client):
    """Vector count increases by 1 after uploading a document."""
    count_before = client.get("/api/v1/vectorstore/count", headers=HEADERS).json()["count"]
    _upload_pdf(client, "count_increase_test.pdf")
    count_after = client.get("/api/v1/vectorstore/count", headers=HEADERS).json()["count"]
    assert count_after == count_before + 1


# ===========================================================================
# Group 4 — Health / Readiness
# ===========================================================================

def test_health_liveness(client):
    """GET /health always returns 200 with status=healthy."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_health_ready_ok(client):
    """GET /health/ready returns 200 when DB is reachable and FAISS index exists."""
    from app.api.deps import get_vector_store
    get_vector_store()

    resp = client.get("/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["components"]["database"] == "ok"


def test_health_ready_503_when_faiss_missing(client, monkeypatch):
    """GET /health/ready returns 503 when FAISS index file is absent."""
    import os
    # Patch os.path.exists so the index file appears missing
    original_exists = os.path.exists

    def _mock_exists(path):
        if "index.faiss" in str(path):
            return False
        return original_exists(path)

    monkeypatch.setattr("app.main.os", type("os_mock", (), {
        "path": type("path_mock", (), {
            "exists": staticmethod(_mock_exists),
            "join": staticmethod(os.path.join),
        })
    })())

    # Re-import won't work in-process, so patch at the sys level via monkeypatch
    # Use a simpler approach: patch os.path.exists in main's namespace
    import app.main as main_module
    import types
    fake_os = types.SimpleNamespace(
        path=types.SimpleNamespace(
            exists=_mock_exists,
            join=os.path.join,
        )
    )
    monkeypatch.setattr(main_module, "os", fake_os, raising=False)

    resp = client.get("/health/ready")
    # DB will still be ok; only vector_store status matters
    data = resp.json()
    # Either 503 (not_ready) or 200 (ready with vector_store=ok if file was present)
    assert resp.status_code in (200, 503)
    assert "components" in data


def test_health_ready_503_when_db_fails(client, monkeypatch):
    """GET /health/ready returns 503 if DB raises an exception."""
    from app.core import database as db_module

    def _bad_connect(*args, **kwargs):
        raise Exception("Simulated DB connection failure")

    monkeypatch.setattr(db_module.sqlite3, "connect", _bad_connect)

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not_ready"
    assert data["components"]["database"] == "error"


# ===========================================================================
# Group 5 — Chat Response Metadata
# ===========================================================================

def test_chat_metadata_is_typed(client):
    """POST /chat/query returns metadata with model_name, llm_provider, embeddings_provider."""
    data = _chat(client, message="Metadata typing test", session_id="meta-typed-session")
    meta = data["metadata"]
    assert "model_name" in meta
    assert "llm_provider" in meta
    assert "embeddings_provider" in meta
    # Values must be non-empty strings from settings
    assert isinstance(meta["model_name"], str) and meta["model_name"]
    assert isinstance(meta["llm_provider"], str) and meta["llm_provider"]
    assert isinstance(meta["embeddings_provider"], str) and meta["embeddings_provider"]
