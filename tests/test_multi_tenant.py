"""
tests/test_multi_tenant.py

Verifies CORS policy changes and multi-tenant per-caller data isolation.
  - CORS credentials must be False when origins are wildcarded.
  - Users with different API keys must not see or modify each other's documents.
  - Users with different API keys must have isolated conversation histories,
    even if they share the same session_id.
  - Multi-key validation (comma-separated keys in configuration) must work correctly.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def configure_multiple_api_keys(monkeypatch):
    """
    Overrides settings.API_KEY to configure two valid keys.
    """
    from app.core.config import settings
    # Set comma-separated valid keys
    monkeypatch.setattr(settings, "API_KEY", "user_a_secret_key, user_b_secret_key")


def test_cors_credentials_false(client):
    """
    Verify that CORS middleware allow_credentials is False.
    """
    # Send a preflight OPTIONS request
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-API-Key",
    }
    response = client.options("/api/v1/chat/query", headers=headers)
    assert response.status_code == 200
    # Credentials must NOT be allowed under a wildcard/open origin setup
    assert response.headers.get("access-control-allow-credentials") is None or response.headers.get("access-control-allow-credentials") == "false"


def test_multiple_api_keys_valid(client):
    """
    Verify that both user_a_secret_key and user_b_secret_key are valid.
    """
    # Try Key A
    resp_a = client.get("/api/v1/documents", headers={"X-API-Key": "user_a_secret_key"})
    assert resp_a.status_code == 200

    # Try Key B
    resp_b = client.get("/api/v1/documents", headers={"X-API-Key": "user_b_secret_key"})
    assert resp_b.status_code == 200

    # Try Invalid Key
    resp_invalid = client.get("/api/v1/documents", headers={"X-API-Key": "invalid_secret_key"})
    assert resp_invalid.status_code == 403


def test_document_isolation(client):
    """
    Verify that documents uploaded by User A cannot be read or deleted by User B.
    """
    headers_a = {"X-API-Key": "user_a_secret_key"}
    headers_b = {"X-API-Key": "user_b_secret_key"}

    # 1. User A uploads a document
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "User A confidential report."
    
    with patch("app.services.pdf_processor.PdfReader") as mock_pdf_reader:
        mock_pdf_reader.return_value.pages = [mock_page]
        files = {"file": ("user_a_doc.pdf", b"%PDF-mock-bytes", "application/pdf")}
        upload_resp = client.post("/api/v1/documents/upload", files=files, headers=headers_a)
        assert upload_resp.status_code == 201
        doc_id = upload_resp.json()["document_id"]

    # 2. User B lists documents — should NOT see User A's document
    list_resp_b = client.get("/api/v1/documents", headers=headers_b)
    assert list_resp_b.status_code == 200
    docs_b = list_resp_b.json()
    assert not any(d["document_id"] == doc_id for d in docs_b)

    # User A lists documents — SHOULD see it
    list_resp_a = client.get("/api/v1/documents", headers=headers_a)
    assert list_resp_a.status_code == 200
    docs_a = list_resp_a.json()
    assert any(d["document_id"] == doc_id for d in docs_a)

    # 3. User B tries to GET the document details by ID — should return 404
    get_resp_b = client.get(f"/api/v1/documents/{doc_id}", headers=headers_b)
    assert get_resp_b.status_code == 404

    # User A gets it — should succeed
    get_resp_a = client.get(f"/api/v1/documents/{doc_id}", headers=headers_a)
    assert get_resp_a.status_code == 200

    # 4. User B tries to DELETE the document — should return 404
    delete_resp_b = client.delete(f"/api/v1/documents/{doc_id}", headers=headers_b)
    assert delete_resp_b.status_code == 404

    # 5. User A deletes it — should succeed
    delete_resp_a = client.delete(f"/api/v1/documents/{doc_id}", headers=headers_a)
    assert delete_resp_a.status_code == 200


def test_chat_session_isolation(client):
    """
    Verify that User A's chat history is isolated from User B's chat history,
    even if they use the same session_id.
    """
    headers_a = {"X-API-Key": "user_a_secret_key"}
    headers_b = {"X-API-Key": "user_b_secret_key"}
    session_id = "shared-session-id-123"

    # User A sends a message in session
    resp_a1 = client.post(
        "/api/v1/chat/query",
        json={"message": "I am User A, secret code 5566", "session_id": session_id},
        headers=headers_a,
    )
    assert resp_a1.status_code == 200

    # User A's session history should have User A's message
    history_resp_a = client.get(f"/api/v1/sessions/{session_id}", headers=headers_a)
    assert history_resp_a.status_code == 200
    messages_a = history_resp_a.json()["messages"]
    assert any("5566" in m["content"] for m in messages_a)

    # User B checks history for same session_id — should return 404 (not found / empty)
    history_resp_b = client.get(f"/api/v1/sessions/{session_id}", headers=headers_b)
    assert history_resp_b.status_code == 404

    # User B queries the session
    resp_b1 = client.post(
        "/api/v1/chat/query",
        json={"message": "I am User B, hello", "session_id": session_id},
        headers=headers_b,
    )
    assert resp_b1.status_code == 200

    # User B's history should contain User B's messages but NOT User A's messages
    history_resp_b2 = client.get(f"/api/v1/sessions/{session_id}", headers=headers_b)
    assert history_resp_b2.status_code == 200
    messages_b = history_resp_b2.json()["messages"]
    assert any("User B" in m["content"] for m in messages_b)
    assert not any("5566" in m["content"] for m in messages_b)
