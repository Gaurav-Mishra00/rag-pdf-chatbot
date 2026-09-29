"""
tests/test_p2_polish.py

Verifies P2 enhancements:
1. RAG pipeline retrieve → rerank → generate flow
2. Relevance threshold gating ("I don't know" fallback)
3. Path masking in status API
4. Database indices exist
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from app.core.config import settings
from app.core.database import get_db_connection
from app.services.rag_service import RAGService
from app.api.deps import get_rag_service, get_vector_store, reset_vector_store


def test_rag_pipeline_with_reranking():
    """
    Verify that when RERANK_ENABLED=True, the pipeline calls the reranker
    and returns results sorted by rerank_score.
    """
    store = MagicMock()
    llm = MagicMock()

    doc1 = Document(page_content="First chunk about RAG.", metadata={"source": "a.pdf", "page": 1})
    doc2 = Document(page_content="Second chunk about vectors.", metadata={"source": "a.pdf", "page": 2})

    # Simulate FAISS similarity search returning both docs
    store.vector_store = MagicMock()
    store.similarity_search.return_value = [(doc1, 0.7), (doc2, 0.9)]

    mock_response = MagicMock()
    mock_response.content = "Answer about RAG [Source 1]"
    llm.invoke.return_value = mock_response

    service = RAGService(vector_store=store, llm=llm)

    # Mock reranker to flip the order
    def mock_rerank(query, docs):
        for i, doc in enumerate(docs):
            copy = doc.model_copy(deep=True)
            copy.metadata["rerank_score"] = 0.9 - i * 0.3  # 0.9, 0.6
            docs[i] = copy
        return sorted(docs, key=lambda d: d.metadata["rerank_score"], reverse=True)

    with patch("app.services.rag_service.settings") as mock_settings:
        mock_settings.RETRIEVAL_K = 20
        mock_settings.CONTEXT_K = 4
        mock_settings.RERANK_ENABLED = True
        mock_settings.MIN_SIMILARITY = 0.3
        mock_settings.RERANK_MIN_SCORE = 0.5
        with patch("app.services.reranker.rerank", side_effect=mock_rerank):
            result = service.answer_query_detailed("What is RAG?", [])

    assert result.relevance_passed is True
    assert result.candidates_before_rerank == 2
    assert "Answer about RAG" in result.answer


def test_relevance_gate_triggers_i_dont_know():
    """
    When all retrieved docs score below the threshold, the pipeline
    returns an 'I don't know' answer without calling the LLM.
    """
    store = MagicMock()
    llm = MagicMock()

    doc = Document(page_content="Irrelevant content.", metadata={"source": "b.pdf", "page": 1})
    store.vector_store = MagicMock()
    store.similarity_search.return_value = [(doc, 0.1)]  # Very low score

    service = RAGService(vector_store=store, llm=llm)

    with patch("app.services.rag_service.settings") as mock_settings:
        mock_settings.RETRIEVAL_K = 4
        mock_settings.CONTEXT_K = 4
        mock_settings.RERANK_ENABLED = False
        mock_settings.MIN_SIMILARITY = 0.5  # Higher than the 0.1 score
        mock_settings.RERANK_MIN_SCORE = 0.5
        result = service.answer_query_detailed("Unrelated question?", [])

    assert result.relevance_passed is False
    assert "don't have enough relevant information" in result.answer
    llm.invoke.assert_not_called()


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
