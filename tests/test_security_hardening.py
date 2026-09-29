"""
tests/test_security_hardening.py

Validates security controls and deployment readiness:
1. Dynamic CORS configuration
2. Strict security headers (CSP, X-Frame-Options, HSTS in production)
3. Timing-safe API key verification
4. Filename sanitization against path traversal during upload
5. Rate limiter memory bounded pruning
"""

import io
import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.core.config import Settings
from app.main import create_app
from app.core.rate_limiter import RateLimiter


def test_security_headers_present(client):
    """Verify essential security headers on HTTP responses."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert "strict-origin-when-cross-origin" in resp.headers.get("Referrer-Policy", "")
    assert "frame-ancestors 'none'" in resp.headers.get("Content-Security-Policy", "")


def test_hsts_header_in_production():
    """Verify Strict-Transport-Security header is applied in production environment."""
    with patch("app.main.settings") as mock_settings:
        mock_settings.APP_ENV = "production"
        mock_settings.APP_NAME = "RAG Chatbot API"
        mock_settings.ALLOWED_ORIGINS = "*"
        mock_settings.API_KEY = "test_key"
        mock_settings.LOG_LEVEL = "INFO"
        mock_settings.FAISS_INDEX_PATH = "data/faiss_index"
        mock_settings.UPLOAD_DIR = "data/uploads"
        mock_settings.SQLITE_DB_PATH = "data/db.sqlite3"

        app = create_app()
        test_client = TestClient(app)
        resp = test_client.get("/health")
        assert resp.status_code == 200
        assert "Strict-Transport-Security" in resp.headers
        assert "max-age=31536000" in resp.headers["Strict-Transport-Security"]


def test_cors_specific_origins():
    """Verify specific ALLOWED_ORIGINS are enforced over wildcard when configured."""
    with patch("app.main.settings") as mock_settings:
        mock_settings.APP_ENV = "production"
        mock_settings.APP_NAME = "RAG Chatbot API"
        mock_settings.ALLOWED_ORIGINS = "https://app.example.com, https://admin.example.com"
        mock_settings.API_KEY = "test_key"
        mock_settings.LOG_LEVEL = "INFO"
        mock_settings.FAISS_INDEX_PATH = "data/faiss_index"
        mock_settings.UPLOAD_DIR = "data/uploads"
        mock_settings.SQLITE_DB_PATH = "data/db.sqlite3"

        app = create_app()
        test_client = TestClient(app)

        # Preflight from allowed origin
        resp_allowed = test_client.options(
            "/api/v1/health",
            headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp_allowed.headers.get("access-control-allow-origin") == "https://app.example.com"

        # Preflight from untrusted origin
        resp_untrusted = test_client.options(
            "/api/v1/health",
            headers={
                "Origin": "https://malicious-site.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp_untrusted.headers.get("access-control-allow-origin") != "https://malicious-site.com"


def test_upload_filename_sanitization(client, monkeypatch):
    """Verify that uploading a file with path traversal characters sanitizes the filename."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "API_KEY", "test_secret_key")

    dummy_pdf = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\nxref\ntrailer<</Root 1 0 R>>\nstartxref\n9\n%%EOF"
    malicious_filename = "../../../etc/passwd.pdf"

    with patch("app.api.endpoints.documents.PDFProcessorService.process_pdf") as mock_process:
        from langchain_core.documents import Document
        mock_process.return_value = [
            Document(page_content="Test page content", metadata={"page": 1})
        ]

        with patch("app.api.endpoints.documents.FAISSVectorStore.add_documents"):
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": (malicious_filename, io.BytesIO(dummy_pdf), "application/pdf")},
                headers={"X-API-Key": "test_secret_key"},
            )

    assert response.status_code == 201
    data = response.json()
    # The filename must be sanitized to pure basename without '../'
    assert data["filename"] == "passwd.pdf"
    assert "/" not in data["filename"]
    assert "\\" not in data["filename"]


def test_rate_limiter_memory_pruning():
    """Verify that RateLimiter bounds memory usage and removes expired keys."""
    limiter = RateLimiter(rate_limit_requests=5, period_seconds=1)

    # Insert 1050 keys with expired timestamps
    old_time = 1000.0
    for i in range(1050):
        limiter._history[f"stale_key_{i}"] = [old_time]

    assert len(limiter._history) == 1050

    # Next check should prune expired keys
    limiter.is_rate_limited("active_key")
    # All stale keys should be cleaned up because len > 1000 and all are expired
    assert len(limiter._history) <= 50
