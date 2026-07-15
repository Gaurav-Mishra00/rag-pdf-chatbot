"""
tests/test_operational.py

Verifies the implementation of production/operational safeguards:
1. Custom in-memory rate limiting (429 status code)
2. Payload constraints (max prompt length validation error)
3. Pagination query params for session and document lists
4. Automated snapshot backups and retention
5. File-based cloud secrets configuration
"""

import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from app.core.config import Settings, resolve_secret_value
from app.core.rate_limiter import chat_limiter, upload_limiter
from app.core.backup import backup_data_assets
from app.core.database import get_db_connection

HEADERS = {"X-API-Key": "test_secret_key"}


def test_rate_limiting_chat(client):
    """
    Verify that sending more than 30 chat requests in under 60 seconds
    correctly triggers a 429 Too Many Requests response.
    """
    chat_limiter.clear()
    from unittest.mock import patch
    
    with patch("app.core.rate_limiter.settings.APP_ENV", "production"):
        # 1. Send 30 successful requests
        for i in range(30):
            # We query status endpoint or a lightweight check, or chat query.
            # Let's mock answer_query to avoid invoking fake LLM 30 times.
            with patch("app.services.rag_service.RAGService.answer_query", return_value=("ans", [])):
                resp = client.post(
                    "/api/v1/chat/query",
                    json={"message": f"test message {i}"},
                    headers=HEADERS,
                )
                assert resp.status_code == 200

        # 2. The 31st request must trigger 429
        with patch("app.services.rag_service.RAGService.answer_query", return_value=("ans", [])):
            resp_blocked = client.post(
                "/api/v1/chat/query",
                json={"message": "blocked query"},
                headers=HEADERS,
            )
            assert resp_blocked.status_code == 429
            assert resp_blocked.json()["detail"] == "Rate limit exceeded. Please try again later."

    # Cleanup
    chat_limiter.clear()


def test_chat_query_message_length_constraint(client):
    """
    Verify that a chat message exceeding 4000 characters
    fails validation (422 Unprocessable Entity).
    """
    oversized_prompt = "a" * 4001
    resp = client.post(
        "/api/v1/chat/query",
        json={"message": oversized_prompt},
        headers=HEADERS,
    )
    assert resp.status_code == 422
    errors = resp.json()["detail"]
    assert any(err["loc"] == ["body", "message"] for err in errors)


def test_pagination_documents_and_sessions(client):
    """
    Verify that document list and session list pagination (limit, offset)
    behaves correctly and returns expected subset of items.
    """
    import hashlib
    hashed_user_id = hashlib.sha256(b"test_secret_key").hexdigest()

    # 1. Populate DB with mock documents
    with get_db_connection() as conn:
        conn.execute("DELETE FROM documents")
        for i in range(5):
            conn.execute(
                """
                INSERT INTO documents (document_id, filename, file_path, file_size_bytes, chunk_count, status, user_id)
                VALUES (?, ?, 'dummy', 100, 1, 'completed', ?)
                """,
                (f"doc-id-{i}", f"doc-name-{i}.pdf", hashed_user_id),
            )

    # 2. Test document list pagination
    # limit = 2
    resp_limit = client.get("/api/v1/documents?limit=2&offset=0", headers=HEADERS)
    assert resp_limit.status_code == 200
    data_limit = resp_limit.json()
    assert len(data_limit) == 2

    # offset = 4, limit = 2 (should return only 1 document remaining)
    resp_offset = client.get("/api/v1/documents?limit=2&offset=4", headers=HEADERS)
    assert resp_offset.status_code == 200
    data_offset = resp_offset.json()
    assert len(data_offset) == 1

    # 3. Populate DB with mock chat history sessions
    with get_db_connection() as conn:
        conn.execute("DELETE FROM chat_history")
        for i in range(5):
            conn.execute(
                "INSERT INTO chat_history (session_id, role, content, user_id) VALUES (?, 'user', 'hi', ?)",
                (f"session-id-{i}", hashed_user_id),
            )

    # 4. Test session list pagination
    # limit = 3
    resp_sess = client.get("/api/v1/sessions?limit=3&offset=0", headers=HEADERS)
    assert resp_sess.status_code == 200
    data_sess = resp_sess.json()
    assert len(data_sess["sessions"]) == 3
    assert data_sess["total"] == 5

    # limit = 3, offset = 3 (returns remaining 2)
    resp_sess_off = client.get("/api/v1/sessions?limit=3&offset=3", headers=HEADERS)
    assert resp_sess_off.status_code == 200
    data_sess_off = resp_sess_off.json()
    assert len(data_sess_off["sessions"]) == 2
    assert data_sess_off["total"] == 5


def test_backup_generation_and_retention():
    """
    Verify that the backup utility correctly bundles assets into zip format
    and prunes older backups to satisfy the retention policy (max_backups).
    """
    with tempfile.TemporaryDirectory() as tmp_backup_dir:
        # Create multiple dummy backups to trigger retention
        # Write dummy files to simulate a history of files
        for i in range(7):
            backup_file = backup_data_assets(backup_dir=tmp_backup_dir, max_backups=5)
            # Sleep briefly to ensure distinct file modification times
            import time
            time.sleep(0.1)

        # Check total remaining files in backup folder
        files = [f for f in os.listdir(tmp_backup_dir) if f.startswith("backup_") and f.endswith(".zip")]
        # Retention limit is max_backups = 5
        assert len(files) == 5


def test_file_secret_resolution():
    """
    Verify that resolving secret values prefixed with file:// reads
    from the target file on disk.
    """
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write("secret-value-extracted-from-file\n")
        tmp_name = tmp.name

    try:
        raw_config_val = f"file://{tmp_name}"
        resolved = resolve_secret_value(raw_config_val)
        assert resolved == "secret-value-extracted-from-file"
    finally:
        os.remove(tmp_name)
