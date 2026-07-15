"""
tests/test_reconciliation.py

Verifies the self-healing database and filesystem reconciliation startup job,
as well as the thread safety locks protecting the diagnostic endpoints.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.core.config import settings
from app.core.database import get_db_connection, reconcile_storage_layers
from app.api.deps import get_vector_store, reset_vector_store


def test_diagnostics_reads_acquire_lock():
    """
    Verify that reading count and index_file_size_bytes properties
    acquires the _faiss_write_lock to ensure read consistency.
    """
    reset_vector_store()
    vector_store = get_vector_store()

    # Verify we can access the lock and mock it
    from app.vectorstore.faiss_store import _faiss_write_lock
    
    mock_lock = MagicMock()
    with patch("app.vectorstore.faiss_store._faiss_write_lock", mock_lock):
        _ = vector_store.count
        _ = vector_store.index_file_size_bytes

    # Both properties must acquire the lock exactly once
    assert mock_lock.__enter__.call_count == 2
    assert mock_lock.__exit__.call_count == 2


def test_reconcile_orphaned_file():
    """
    Verify that an untracked PDF file in settings.UPLOAD_DIR
    (with no database entry) is deleted during reconciliation.
    """
    # 1. Create a dummy untracked file
    doc_id = "reconcile-orphan-file-id"
    file_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}.pdf")
    
    with open(file_path, "wb") as f:
        f.write(b"%PDF-untracked-file-content")

    assert os.path.exists(file_path) is True

    # 2. Run reconciliation
    reset_vector_store()
    vector_store = get_vector_store()
    reconcile_storage_layers(vector_store, settings.UPLOAD_DIR)

    # 3. File must be deleted
    assert os.path.exists(file_path) is False


def test_reconcile_orphaned_db_record():
    """
    Verify that a database record referencing chunk IDs that do NOT
    exist in the FAISS index is purged during reconciliation.
    """
    doc_id = "reconcile-orphan-db-doc"
    file_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}.pdf")
    
    # 1. Write dummy physical file so it doesn't fail file checks during DB purge
    with open(file_path, "wb") as f:
        f.write(b"%PDF-content")

    # 2. Insert record in SQLite documents and document_chunks
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO documents (document_id, filename, file_path, file_size_bytes, chunk_count, status, user_id)
            VALUES (?, 'orphan_doc.pdf', ?, 100, 1, 'completed', 'default_user')
            """,
            (doc_id, file_path),
        )
        conn.execute(
            "INSERT INTO document_chunks (chunk_id, document_id) VALUES ('orphan-chunk-id-123', ?)",
            (doc_id,),
        )

    # Verify records exist
    with get_db_connection() as conn:
        row = conn.execute("SELECT 1 FROM documents WHERE document_id = ?", (doc_id,)).fetchone()
        assert row is not None

    # 3. Run reconciliation (since FAISS index does not contain 'orphan-chunk-id-123')
    reset_vector_store()
    vector_store = get_vector_store()
    reconcile_storage_layers(vector_store, settings.UPLOAD_DIR)

    # 4. Database record and physical file must be purged
    with get_db_connection() as conn:
        row_after = conn.execute("SELECT 1 FROM documents WHERE document_id = ?", (doc_id,)).fetchone()
        assert row_after is None
        chunk_after = conn.execute("SELECT 1 FROM document_chunks WHERE document_id = ?", (doc_id,)).fetchone()
        assert chunk_after is None

    assert os.path.exists(file_path) is False


def test_reconcile_orphaned_faiss_chunks():
    """
    Verify that vectors/chunks in FAISS that have no database tracking record
    are purged from the FAISS index during reconciliation.
    """
    reset_vector_store()
    vector_store = get_vector_store()
    initial_count = vector_store.count

    # 1. Add chunks to FAISS directly without database entries
    orphan_chunk_id = "orphan-faiss-chunk-999"
    doc = Document(page_content="Orphaned FAISS content.", metadata={"source": "orphan_vector.pdf", "page": 1})
    vector_store.add_documents([doc], ids=[orphan_chunk_id])
    
    assert vector_store.count == initial_count + 1

    # 2. Run reconciliation
    reconcile_storage_layers(vector_store, settings.UPLOAD_DIR)

    # 3. The orphaned chunk must be removed from FAISS
    assert vector_store.count == initial_count
