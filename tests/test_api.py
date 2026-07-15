from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """
    Test the healthcheck endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_query_unauthorized(client: TestClient):
    """
    Test that endpoints require API key authentication.
    """
    response = client.post("/api/v1/chat/query", json={"message": "hello"})
    assert response.status_code == 401


def test_chat_query_authorized(client: TestClient):
    """
    Test chat query with proper authorization.
    """
    headers = {"X-API-Key": "test_secret_key"}
    response = client.post(
        "/api/v1/chat/query",
        json={"message": "What is in the document?"},
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["session_id"] == "default-session"


def test_document_upload_requires_pdf(client: TestClient):
    """
    Test that the upload endpoint rejects non-PDFs.
    """
    headers = {"X-API-Key": "test_secret_key"}
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    response = client.post("/api/v1/documents/upload", files=files, headers=headers)
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_vectorstore_status(client: TestClient):
    """
    Test vectorstore status endpoint.
    """
    headers = {"X-API-Key": "test_secret_key"}
    response = client.get("/api/v1/vectorstore/status", headers=headers)
    assert response.status_code == 200
    assert "status" in response.json()


def test_chat_query_history_persists(client: TestClient):
    """
    Test that sending multiple queries under the same session_id 
    correctly populates and updates the conversational history.
    """
    import anyio
    from app.api.deps import get_history_manager
    
    headers = {"X-API-Key": "test_secret_key"}
    session_id = "test-session-123"
    
    history_manager = get_history_manager()
    anyio.run(history_manager.clear_history, session_id)
    assert len(anyio.run(history_manager.get_history, session_id)) == 0

    # Send first message
    response = client.post(
        "/api/v1/chat/query",
        json={"message": "First message", "session_id": session_id},
        headers=headers,
    )
    assert response.status_code == 200
    
    # Verify history has 2 entries (user + assistant)
    history = anyio.run(history_manager.get_history, session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "First message"
    assert history[1]["role"] == "assistant"
    
    # Send second message
    response2 = client.post(
        "/api/v1/chat/query",
        json={"message": "Second message", "session_id": session_id},
        headers=headers,
    )
    assert response2.status_code == 200
    
    # Verify history has 4 entries
    history_after = anyio.run(history_manager.get_history, session_id)
    assert len(history_after) == 4
    assert history_after[2]["content"] == "Second message"
