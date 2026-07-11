import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from app.core.config import settings


def test_e2e_upload_and_chat(client: TestClient, monkeypatch, tmp_path):
    """
    End-to-End test that validates the entire RAG pipeline:
    1. Uploads a mock PDF document.
    2. Verifies that PDF pages are split, indexed, and stored in FAISS.
    3. Checks the vector store status endpoint.
    4. Queries the chatbot and asserts the correct source document,
       page number, and similarity score are returned in the citations.
    """
    # 1. Override the local FAISS index path to a temporary path for isolation
    test_index_path = os.path.join(tmp_path, "faiss_index")
    monkeypatch.setattr(settings, "FAISS_INDEX_PATH", test_index_path)

    headers = {"X-API-Key": "test_secret_key"}

    # 2. Mock PdfReader to return custom document content
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = (
        "FastAPI is a modern, fast (high-performance) web framework for building APIs."
    )
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = (
        "FAISS is a library for efficient similarity search and clustering of dense vectors."
    )

    with patch("app.services.pdf_processor.PdfReader") as mock_pdf_reader:
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader_instance

        # 3. Perform file upload
        files = {"file": ("manual.pdf", b"%PDF-mock-bytes", "application/pdf")}
        upload_resp = client.post(
            "/api/v1/documents/upload", files=files, headers=headers
        )
        assert upload_resp.status_code == 201
        upload_data = upload_resp.json()
        assert upload_data["status"] == "completed"
        assert "Successfully parsed" in upload_data["message"]

    # 4. Check vectorstore status
    status_resp = client.get("/api/v1/vectorstore/status", headers=headers)
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "ready"
    assert status_resp.json()["has_local_index"] is True

    # 5. Submit query to chatbot about the second page (FAISS library)
    query_resp = client.post(
        "/api/v1/chat/query",
        json={"message": "What is FAISS?", "session_id": "e2e-session-id"},
        headers=headers,
    )
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    assert "answer" in query_data
    assert query_data["session_id"] == "e2e-session-id"

    # 6. Verify citations contain the exact page matching the query
    sources = query_data["sources"]
    assert len(sources) > 0

    match_found = False
    for source in sources:
        if "FAISS is a library" in source["snippet"]:
            assert source["document_name"] == "manual.pdf"
            assert source["page"] == 2
            assert source["score"] is not None
            match_found = True

    assert match_found, "Matching source page document was not found in citations"
