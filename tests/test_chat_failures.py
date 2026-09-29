from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from app.api.deps import get_llm, get_embeddings, get_rag_service
from app.core.config import settings
from app.main import app
from app.services.errors import GenerationError, generation_error


@pytest.mark.parametrize("error, status", [
    (httpx.ReadTimeout("secret-provider-details"), 504),
    (httpx.ConnectError("secret-provider-details"), 503),
    (RuntimeError("secret-provider-details"), 502),
])
def test_provider_errors_are_safe(error, status):
    result = generation_error(error)
    assert result.status_code == status
    assert "secret-provider-details" not in str(result)


def test_provider_server_error_is_actionable():
    error = RuntimeError("Internal error encountered")
    error.code = 500
    assert "AI provider returned a server error" in str(generation_error(error))


def test_generation_failure_does_not_save_a_turn(client):
    from app.services.rag_service import RAGResult
    service = MagicMock()
    service.answer_query_detailed.side_effect = GenerationError("AI service timed out", 504)
    app.dependency_overrides[get_rag_service] = lambda: service
    headers = {"X-API-Key": "test_secret_key"}
    try:
        response = client.post("/api/v1/chat/query", headers=headers,
                               json={"message": "what is machine learning", "session_id": "failed-generation"})
        assert response.status_code == 504
        assert response.json() == {"detail": "AI service timed out"}
        history = client.get("/api/v1/sessions/failed-generation", headers=headers)
        assert history.status_code == 404

        # Retrying the same session after provider recovery creates only a real turn.
        service.answer_query_detailed.side_effect = None
        service.answer_query_detailed.return_value = RAGResult(
            answer="Machine learning learns patterns from data.",
            relevance_passed=True,
        )
        response = client.post("/api/v1/chat/query", headers=headers,
                               json={"message": "what is machine learning", "session_id": "failed-generation"})
        assert response.status_code == 200
        history = client.get("/api/v1/sessions/failed-generation", headers=headers).json()
        assert history["total"] == 2
        assert history["messages"][1]["content"] == response.json()["answer"]
    finally:
        app.dependency_overrides.pop(get_rag_service, None)


@pytest.mark.parametrize("factory", [get_llm, get_embeddings])
def test_missing_provider_key_never_silently_uses_mock(monkeypatch, factory):
    monkeypatch.setattr(settings, "APP_ENV", "local")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    with pytest.raises(HTTPException) as error:
        factory()
    assert error.value.status_code == 503


def test_empty_model_response_is_not_an_answer():
    from app.services.rag_service import RAGService
    from langchain_core.documents import Document
    store = MagicMock()
    store.vector_store = MagicMock()
    doc = Document(page_content="content", metadata={"source": "test.pdf", "page": 1})
    store.similarity_search.return_value = [(doc, 0.9)]

    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "  "
    llm.invoke.return_value = mock_response

    service = RAGService(store, llm)
    with patch("app.services.rag_service.settings") as mock_settings:
        mock_settings.RETRIEVAL_K = 4
        mock_settings.CONTEXT_K = 4
        mock_settings.RERANK_ENABLED = False
        mock_settings.MIN_SIMILARITY = 0.3
        mock_settings.RERANK_MIN_SCORE = 0.5
        with pytest.raises(GenerationError, match="empty answer"):
            service.answer_query("hello", [])
