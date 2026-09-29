from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from unittest.mock import patch

import pytest

from app.api.deps import get_embeddings, reset_vector_store
from app.core.config import settings


def test_embedding_model_constructed_once_for_concurrent_requests(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDINGS_PROVIDER", "huggingface")
    with patch("langchain_huggingface.HuggingFaceEmbeddings") as factory:
        with ThreadPoolExecutor(max_workers=8) as pool:
            instances = list(pool.map(lambda _: get_embeddings(), range(24)))
        factory.assert_called_once_with(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True, "batch_size": settings.EMBEDDING_BATCH_SIZE},
        )
        assert all(instance is instances[0] for instance in instances)
        assert get_embeddings() is instances[0]
        reset_vector_store()
        get_embeddings()
        assert factory.call_count == 2


class ScriptParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.scripts.append(dict(attrs))


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/docs/oauth2-redirect"])
def test_docs_scripts_have_matching_csp_nonce(client, path):
    response = client.get(path)
    assert response.status_code == 200
    policy = response.headers["Content-Security-Policy"]
    script_policy = next(part for part in policy.split(";") if "script-src" in part)
    assert "https://cdn.jsdelivr.net" in script_policy
    assert "'unsafe-inline'" not in script_policy
    parser = ScriptParser()
    parser.feed(response.text)
    assert parser.scripts
    for script in parser.scripts:
        assert script["nonce"]
        assert f"'nonce-{script['nonce']}'" in script_policy
    second_response = client.get(path)
    assert second_response.headers["Content-Security-Policy"] != policy
    assert response.headers["Cache-Control"] == "no-store"


def test_docs_policy_does_not_relax_main_app(client):
    policy = client.get("/").headers["Content-Security-Policy"]
    assert "default-src 'self'" in policy
    assert "cdn.jsdelivr.net" not in policy
    assert "nonce-" not in policy
    schema = client.get("/openapi.json").json()
    assert "/api/v1/chat/query" in schema["paths"]
    assert "/docs" not in schema["paths"]
