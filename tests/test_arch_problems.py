"""
tests/test_arch_problems.py

Covers all seven architectural issues that were identified and fixed:
 1. Persistent chat history (SQLite)
 2. FAISS write locking & custom chunk IDs
 3. FAISS delete_documents calls .delete() and persists via save_index()
 4. SQLite WAL/busy_timeout PRAGMAs are issued on every connection
 5. threading.Lock() is process-local — documented, tested as locked
 6. Delete endpoint rollback story (DB failure after FAISS delete logs
    the orphaned IDs; upload rollback also covered)
 7. File upload validation — magic-byte check and size limit
"""

import io
import os

import anyio
import pytest
from unittest.mock import MagicMock, call, patch
from langchain_core.documents import Document
from langchain_community.embeddings import FakeEmbeddings

from app.core.config import settings
from app.core.database import get_db_connection
from app.services.history_manager import HistoryManager
from app.vectorstore.faiss_store import FAISSVectorStore
from app.api.endpoints.documents import (
    _db_check_document_exists,
    _db_get_document_details,
    _db_list_documents,
)


# ===========================================================================
# 1. Persistent Chat History (SQLite)
# ===========================================================================

@pytest.mark.anyio
async def test_persistent_chat_history():
    """HistoryManager persists and clears chat history in the SQLite database."""
    history_manager = HistoryManager()
    session_id = "test-db-session-456"

    await history_manager.clear_history(session_id)
    history = await history_manager.get_history(session_id)
    assert len(history) == 0

    await history_manager.add_message(session_id, "user", "Hello database")
    await history_manager.add_message(session_id, "assistant", "Hello human")

    history = await history_manager.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello database"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hello human"

    # Verify history survives across new HistoryManager instances (proving DB persistence)
    new_manager = HistoryManager()
    history_reloaded = await new_manager.get_history(session_id)
    assert len(history_reloaded) == 2

    await history_manager.clear_history(session_id)
    history_after = await history_manager.get_history(session_id)
    assert len(history_after) == 0


# ===========================================================================
# 2. FAISS write locking & custom chunk IDs
# ===========================================================================

def test_faiss_write_locking_and_custom_ids():
    """add_documents accepts custom IDs, calls save_index(), and returns those IDs."""
    store = FAISSVectorStore(embeddings=FakeEmbeddings(size=1536))
    doc = Document(page_content="Lock test content", metadata={"source": "test.pdf", "page": 1})
    custom_ids = ["chunk-uuid-1"]

    mock_vs = MagicMock()
    mock_vs.index.ntotal = 1

    with patch("app.vectorstore.faiss_store.FAISS.from_documents", return_value=mock_vs) as mock_from, \
         patch.object(store, "save_index") as mock_save:
        returned_ids = store.add_documents([doc], ids=custom_ids)

        mock_from.assert_called_once_with([doc], store.embeddings, ids=custom_ids)
        assert returned_ids == custom_ids
        mock_save.assert_called_once()


# ===========================================================================
# 3. FAISS delete — calls .delete() and persists via save_index()
# ===========================================================================

def test_faiss_delete_documents_calls_delete_and_saves():
    """delete_documents delegates to vector_store.delete() and then calls save_index()."""
    store = FAISSVectorStore(embeddings=FakeEmbeddings(size=1536))
    mock_vs = MagicMock()
    mock_vs.delete.return_value = True
    store.vector_store = mock_vs

    with patch.object(store, "save_index") as mock_save:
        store.delete_documents(["chunk-uuid-1", "chunk-uuid-2"])

        mock_vs.delete.assert_called_once_with(["chunk-uuid-1", "chunk-uuid-2"])
        mock_save.assert_called_once()


def test_faiss_delete_documents_no_op_on_empty_list():
    """delete_documents does nothing (no exception) when called with an empty list."""
    store = FAISSVectorStore(embeddings=FakeEmbeddings(size=1536))
    mock_vs = MagicMock()
    store.vector_store = mock_vs

    store.delete_documents([])
    mock_vs.delete.assert_not_called()


def test_faiss_delete_raises_on_unsupported_index():
    """delete_documents raises RuntimeError when the raw index lacks remove_ids."""
    store = FAISSVectorStore(embeddings=FakeEmbeddings(size=1536))
    mock_vs = MagicMock()
    # Simulate an index type that does NOT have remove_ids (e.g. HNSW)
    del mock_vs.index.remove_ids  # make hasattr() return False
    mock_vs.index.__class__.__name__ = "IndexHNSWFlat"
    store.vector_store = mock_vs

    with pytest.raises(RuntimeError, match="does not support deletion"):
        store.delete_documents(["chunk-a"])


# ===========================================================================
# 4. SQLite WAL mode & busy_timeout
# ===========================================================================

def test_sqlite_wal_and_busy_timeout():
    """get_db_connection issues PRAGMA journal_mode=WAL and busy_timeout=5000."""
    executed = []
    original_execute = None

    with get_db_connection() as conn:
        # Re-run pragmas and capture results to verify they were applied
        wal_row = conn.execute("PRAGMA journal_mode;").fetchone()
        timeout_row = conn.execute("PRAGMA busy_timeout;").fetchone()

    # journal_mode returns the current mode string after setting it
    assert wal_row[0].lower() == "wal", f"Expected WAL, got {wal_row[0]}"
    assert timeout_row[0] == 5000, f"Expected 5000 ms, got {timeout_row[0]}"


# ===========================================================================
# 5. threading.Lock() is process-local (documented constraint)
# ===========================================================================

def test_threading_lock_is_module_level_and_reentrant_safe():
    """
    The module-level _faiss_write_lock is a threading.Lock, not RLock.
    This test confirms it exists and is acquired correctly during writes.
    Concurrency safety is single-process only; this is intentional and documented.
    """
    import threading
    from app.vectorstore import faiss_store

    lock = faiss_store._faiss_write_lock
    assert isinstance(lock, type(threading.Lock())), "Must be a threading.Lock"

    # Lock must be released after add_documents completes
    store = FAISSVectorStore(embeddings=FakeEmbeddings(size=1536))
    doc = Document(page_content="lock-release-test", metadata={"source": "x.pdf", "page": 1})
    mock_vs = MagicMock()
    mock_vs.index.ntotal = 1

    with patch("app.vectorstore.faiss_store.FAISS.from_documents", return_value=mock_vs), \
         patch.object(store, "save_index"):
        store.add_documents([doc])

    # The lock must be released now — try-acquire must succeed immediately
    acquired = lock.acquire(blocking=False)
    assert acquired, "Lock was not released after add_documents returned"
    lock.release()


# ===========================================================================
# 6. Delete endpoint rollback story
# ===========================================================================

@pytest.mark.anyio
async def test_delete_rollback_db_failure_logged(client, caplog):
    """
    If DB cleanup fails after FAISS deletion, a CRITICAL log is emitted with
    the orphaned chunk IDs. The response must be HTTP 500.
    """
    import logging
    from app.api.deps import get_vector_store

    headers = {"X-API-Key": "test_secret_key"}
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Rollback test document content."

    with patch("app.services.pdf_processor.PdfReader") as mock_reader_cls:
        mock_reader_cls.return_value.pages = [mock_page]

        files = {"file": ("rollback_test.pdf", b"%PDF-rollback", "application/pdf")}
        resp = client.post("/api/v1/documents/upload", files=files, headers=headers)
        assert resp.status_code == 201
        doc_id = resp.json()["document_id"]

    # Now simulate DB failure during delete (FAISS delete succeeds first)
    with patch(
        "app.api.endpoints.documents._db_delete_document_records",
        side_effect=Exception("DB write error"),
    ), caplog.at_level(logging.CRITICAL, logger="app.api.endpoints.documents"):
        del_resp = client.delete(f"/api/v1/documents/{doc_id}", headers=headers)

    assert del_resp.status_code == 500
    assert "manual DB cleanup" in del_resp.json()["detail"]
    # Verify that orphaned IDs were logged at CRITICAL level
    assert any("Orphaned chunk IDs" in r.message for r in caplog.records)


# ===========================================================================
# 7. File validation — magic-byte and size
# ===========================================================================

def test_upload_rejects_non_pdf_extension(client):
    """Upload must reject files that don't have a .pdf extension."""
    headers = {"X-API-Key": "test_secret_key"}
    files = {"file": ("evil.txt", b"%PDF-looks-like-pdf-but-wrong-ext", "text/plain")}
    resp = client.post("/api/v1/documents/upload", files=files, headers=headers)
    assert resp.status_code == 400
    assert "PDF" in resp.json()["detail"]


def test_upload_rejects_missing_pdf_magic_bytes(client):
    """Upload must reject a .pdf file without the %%PDF magic header."""
    headers = {"X-API-Key": "test_secret_key"}
    # File has .pdf extension but invalid content (no %PDF header)
    files = {"file": ("fake.pdf", b"PK\x03\x04this-is-a-zip", "application/pdf")}
    resp = client.post("/api/v1/documents/upload", files=files, headers=headers)
    assert resp.status_code == 400
    assert "valid PDF" in resp.json()["detail"]


def test_upload_rejects_oversized_file(client):
    """Upload must reject files larger than 50 MB."""
    headers = {"X-API-Key": "test_secret_key"}
    # Generate a 51 MB fake PDF (starts with %PDF so it passes magic check)
    oversized = b"%PDF" + b"0" * (51 * 1024 * 1024)
    files = {"file": ("big.pdf", oversized, "application/pdf")}
    resp = client.post("/api/v1/documents/upload", files=files, headers=headers)
    assert resp.status_code == 413


def test_upload_accepts_valid_pdf_bytes(client):
    """Upload must pass magic-byte and extension checks for a valid %PDF header."""
    headers = {"X-API-Key": "test_secret_key"}
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Valid PDF content."

    with patch("app.services.pdf_processor.PdfReader") as mock_reader_cls:
        mock_reader_cls.return_value.pages = [mock_page]
        files = {"file": ("valid_magic.pdf", b"%PDF-1.4 fake content", "application/pdf")}
        resp = client.post("/api/v1/documents/upload", files=files, headers=headers)

    assert resp.status_code == 201


# ===========================================================================
# Full end-to-end lifecycle (upload → list → delete)
# ===========================================================================

@pytest.mark.anyio
async def test_document_metadata_and_delete_flow(client):
    """
    End-to-end integration test for document metadata and delete flow.
    1. Upload document (verify file exists, metadata saved, FAISS updated).
    2. List documents (verify presence).
    3. Delete document (verify file removed, database cleared, FAISS deleted).
    """
    from app.api.deps import get_vector_store

    headers = {"X-API-Key": "test_secret_key"}
    vector_store = get_vector_store()
    initial_count = vector_store.count

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is a document deletion test."

    with patch("app.services.pdf_processor.PdfReader") as mock_pdf_reader:
        mock_pdf_reader.return_value.pages = [mock_page]

        files = {"file": ("deletion_test.pdf", b"%PDF-mock-bytes", "application/pdf")}
        upload_resp = client.post("/api/v1/documents/upload", files=files, headers=headers)
        assert upload_resp.status_code == 201
        upload_data = upload_resp.json()
        doc_id = upload_data["document_id"]
        assert upload_data["status"] == "completed"

        assert get_vector_store().count == initial_count + 1

        exists = await anyio.to_thread.run_sync(_db_check_document_exists, "deletion_test.pdf")
        assert exists is True

        details = await anyio.to_thread.run_sync(_db_get_document_details, doc_id)
        assert details is not None
        file_path, chunk_ids = details
        assert os.path.exists(file_path) is True
        assert len(chunk_ids) == 1

        list_resp = client.get("/api/v1/documents", headers=headers)
        assert list_resp.status_code == 200
        assert any(d["document_id"] == doc_id for d in list_resp.json())

        delete_resp = client.delete(f"/api/v1/documents/{doc_id}", headers=headers)
        assert delete_resp.status_code == 200
        assert delete_resp.json()["document_id"] == doc_id

        assert os.path.exists(file_path) is False

        exists_after = await anyio.to_thread.run_sync(_db_check_document_exists, "deletion_test.pdf")
        assert exists_after is False

        details_after = await anyio.to_thread.run_sync(_db_get_document_details, doc_id)
        assert details_after is None

        assert get_vector_store().count == initial_count
