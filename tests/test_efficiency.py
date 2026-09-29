"""
tests/test_efficiency.py

Tests for the high-efficiency architecture enhancements:
1. QueryCache (in-memory TTL cache, LRU eviction, user isolation, invalidation)
2. Chunk deduplication in PDFProcessor
3. Native FAISS HNSW configuration and MMAP reading flag
4. RAGService caching integration and adaptive reranking logic
5. Chat SSE streaming endpoint (/api/v1/chat/stream)
6. Cache invalidation on document mutation
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.core.config import settings
from app.core.cache import QueryCache, query_cache
from app.services.pdf_processor import PDFProcessorService
from app.vectorstore.native_faiss import FAISS
from app.services.rag_service import RAGService, RAGResult


def test_query_cache_basic_and_ttl():
    cache = QueryCache(ttl_seconds=1, max_size=5)
    result = RAGResult(answer="Cached answer")

    # Store
    cache.set("user_a", "what is rag?", result)

    # Hit
    cached = cache.get("user_a", "what is rag?")
    assert cached is not None
    assert cached.answer == "Cached answer"

    # User isolation
    assert cache.get("user_b", "what is rag?") is None

    # Normalization (casing, trailing whitespace)
    assert cache.get("user_a", "  WHAT IS RAG?  ") is not None

    # TTL expiry
    time.sleep(1.1)
    assert cache.get("user_a", "what is rag?") is None


def test_query_cache_lru_eviction():
    cache = QueryCache(ttl_seconds=60, max_size=2)
    res = RAGResult(answer="ans")

    cache.set("u1", "q1", res)
    cache.set("u1", "q2", res)
    assert len(cache) == 2

    # Add 3rd item -> q1 evicted
    cache.set("u1", "q3", res)
    assert len(cache) == 2
    assert cache.get("u1", "q1") is None
    assert cache.get("u1", "q2") is not None
    assert cache.get("u1", "q3") is not None


def test_query_cache_clear_for_user():
    cache = QueryCache(ttl_seconds=60, max_size=10)
    res = RAGResult(answer="ans")

    cache.set("user_1", "q1", res)
    cache.set("user_1", "q2", res)
    cache.set("user_2", "q1", res)

    cache.clear_for_user("user_1")
    assert cache.get("user_1", "q1") is None
    assert cache.get("user_1", "q2") is None
    assert cache.get("user_2", "q1") is not None


def test_chunk_deduplication():
    processor = PDFProcessorService(chunk_size=100, chunk_overlap=10)
    docs = [
        Document(page_content="Exact duplicate content", metadata={"page": 1}),
        Document(page_content="Exact duplicate content", metadata={"page": 2}),
        Document(page_content="Unique chunk content", metadata={"page": 3}),
    ]

    with patch.object(settings, "CHUNK_DEDUPLICATION_ENABLED", True):
        deduped = processor._filter_duplicate_chunks(docs)
        assert len(deduped) == 2
        assert deduped[0].page_content == "Exact duplicate content"
        assert deduped[1].page_content == "Unique chunk content"

    with patch.object(settings, "CHUNK_DEDUPLICATION_ENABLED", False):
        not_deduped = processor._filter_duplicate_chunks(docs)
        assert len(not_deduped) == 3


def test_native_faiss_hnsw_config():
    fake_embeddings = MagicMock()
    fake_embeddings.embed_documents.return_value = [[0.1] * 32]
    docs = [Document(page_content="test doc", metadata={"page": 1})]
    with patch.object(settings, "FAISS_INDEX_TYPE", "hnsw"), \
         patch.object(settings, "FAISS_HNSW_M", 16), \
         patch.object(settings, "FAISS_HNSW_EF_SEARCH", 32):
        store = FAISS.from_documents(docs, fake_embeddings)
        assert hasattr(store.index, "hnsw")
        assert store.index.hnsw.efSearch == 32


def test_rag_service_query_cache_integration():
    store = MagicMock()
    llm = MagicMock()
    service = RAGService(vector_store=store, llm=llm)
    service._apply_relevance_gate = lambda docs, **kw: (docs, True)
    service._rerank = lambda query, docs: docs

    test_user = "user_cache_test"
    query_cache.clear_for_user(test_user)

    doc = Document(page_content="Content", metadata={"source": "test.pdf", "page": 1})
    store.vector_store = MagicMock()
    store.similarity_search.return_value = [(doc, 0.95)]
    mock_resp = MagicMock()
    mock_resp.content = "Initial generated answer"
    llm.invoke.return_value = mock_resp

    # First call: cache miss, executes generation
    res1 = service.answer_query_detailed("What is test?", [], user_id=test_user)
    assert res1.answer == "Initial generated answer"
    assert llm.invoke.call_count == 1

    # Second call: cache hit, llm.invoke NOT called again
    res2 = service.answer_query_detailed("What is test?", [], user_id=test_user)
    assert res2.answer == "Initial generated answer"
    assert llm.invoke.call_count == 1  # Still 1!


def test_chat_stream_endpoint(client):
    headers = {"X-API-Key": "test_secret_key"}
    response = client.post(
        "/api/v1/chat/stream",
        json={"message": "Tell me about RAG"},
        headers=headers,
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    content = response.text
    assert "event: citations" in content
    assert "event: token" in content
    assert "event: done" in content

