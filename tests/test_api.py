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
