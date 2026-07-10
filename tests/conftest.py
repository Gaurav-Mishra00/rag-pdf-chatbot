import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

# Force override test credentials
settings.API_KEY = "test_secret_key"
settings.APP_ENV = "testing"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """
    Fixture providing a test client configured to point to the FastAPI app.
    """
    with TestClient(app) as test_client:
        yield test_client
