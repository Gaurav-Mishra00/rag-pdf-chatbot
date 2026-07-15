"""
tests/test_p2_polish.py

Verifies P2 enhancements:
1. Reentrant caching of RAGService chain
2. Similarity score lookup avoids duplicate string collisions by matching metadata
3. Path masking in status API
4. Database indices exist
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.core.config import settings
from app.core.database import get_db_connection
from app.services.rag_service import RAGService
from app.api.deps import get_rag_service, get_vector_store, reset_vector_store


def test_rag_service_chain_caching():
    """
    Verify RAGService caches the compiled chain and only rebuilds
    it if the underlying FAISS index instance changes.
    """
    reset_vector_store()
    vector_store = get_vector_store()
    
    # We must load/init an index to allow _build_chain to succeed
    vector_store.create_empty_index()

    llm = MagicMock()
    rag_service = RAGService(vector_store=vector_store, llm=llm)

    # Mock the internal chain invoke and similarity_search methods to prevent execution errors
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "answer": "cached answer",
        "context": []
    }
    
    # Spy on _build_chain
    original_build_chain = rag_service._build_chain
    build_chain_spy = MagicMock(side_effect=original_build_chain)
    rag_service._build_chain = build_chain_spy

    # First query -> builds the chain
    with patch.object(rag_service, "_chain", mock_chain):
        # We manually set _chain to None so it triggers the build_chain call
        rag_service._chain = None
        rag_service.answer_query("query 1", [])
        assert build_chain_spy.call_count == 1
        
        # Second query -> uses cached chain, build_chain NOT called again
        rag_service.answer_query("query 2", [])
        assert build_chain_spy.call_count == 1

        # Change underlying vector store index instance
        vector_store.vector_store = MagicMock()
        # Third query -> detects change, clears cache, and attempts to rebuild (catches error and returns string)
        rag_service.answer_query("query 3", [])
        assert build_chain_spy.call_count == 2


def test_similarity_scores_no_duplicate_collision():
    """
    Verify that similarity search score mapping uses metadata comparison
    to prevent collision when multiple chunks share the same content.
    """
    vector_store = MagicMock()
    # Mock similarity search to return two documents with the exact same content but different metadata/pages
    doc1 = Document(page_content="duplicate header content", metadata={"source": "a.pdf", "page": 1})
    doc2 = Document(page_content="duplicate header content", metadata={"source": "a.pdf", "page": 2})

    vector_store.similarity_search.return_value = [
        (doc1, 0.95),
        (doc2, 0.60),
    ]

    llm = MagicMock()
    rag_service = RAGService(vector_store=vector_store, llm=llm)
    
    # Mock chain execution output
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "answer": "answer",
        "context": [doc1, doc2]
    }
    rag_service._chain = mock_chain
    rag_service._vector_store_underlying = vector_store.vector_store

    # Answer query
    _, source_docs = rag_service.answer_query("test query", [])

    # doc1 and doc2 must preserve their respective scores (0.95 vs 0.60)
    assert source_docs[0].metadata["score"] == 0.95
    assert source_docs[1].metadata["score"] == 0.60


def test_status_endpoint_masks_absolute_path(client):
    """
    Verify that the GET /vectorstore/status endpoint does NOT disclose
    the raw absolute directory path and only returns the folder basename.
    """
    HEADERS = {"X-API-Key": "test_secret_key"}
    resp = client.get("/api/v1/vectorstore/status", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "index_path" in data
    # Path must be masked to just the basename
    assert data["index_path"] == os.path.basename(settings.FAISS_INDEX_PATH)
    assert "/" not in data["index_path"]
    assert "\\" not in data["index_path"]


def test_database_indices_exist():
    """
    Verify that performance indices are created in the SQLite database.
    """
    with get_db_connection() as conn:
        # Check index list in database
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        indices = [row["name"] for row in cursor.fetchall()]

    assert "idx_chat_history_session_user" in indices
    assert "idx_document_chunks_doc_id" in indices
    assert "idx_documents_user" in indices
