import os
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient
from app.core.config import settings
from app.core.database import init_db

# Create an isolated temporary directory for all test storage
TEST_TEMP_DIR = tempfile.mkdtemp()

# Force override configuration for isolated test execution
settings.API_KEY = "test_secret_key"
settings.APP_ENV = "testing"
# Unit/integration tests must not use credentials or providers from the local .env.
settings.LLM_PROVIDER = "openai"
settings.EMBEDDINGS_PROVIDER = "openai"
settings.OPENAI_API_KEY = None
settings.GOOGLE_API_KEY = None
settings.ANTHROPIC_API_KEY = None
settings.SQLITE_DB_PATH = os.path.join(TEST_TEMP_DIR, "test_db.sqlite3")
settings.UPLOAD_DIR = os.path.join(TEST_TEMP_DIR, "test_uploads")
settings.FAISS_INDEX_PATH = os.path.join(TEST_TEMP_DIR, "test_faiss_index")

# Initialize test database and local storage structures
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
init_db()

# Import app after configuration is overridden to ensure lifespan runs with test settings
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_dir():
    """
    Autouse session fixture to purge the temporary test directory
    once all tests complete.
    """
    yield
    try:
        shutil.rmtree(TEST_TEMP_DIR)
    except Exception:
        pass


@pytest.fixture(scope="module")
def client() -> TestClient:
    """
    Fixture providing a test client configured to point to the FastAPI app.
    """
    from app.api.deps import get_embeddings, get_llm
    from langchain_core.embeddings import FakeEmbeddings
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_embeddings] = lambda: FakeEmbeddings(size=1536)
    app.dependency_overrides[get_llm] = lambda: FakeListChatModel(responses=["Test answer"])
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(overrides)


@pytest.fixture(autouse=True)
def reset_singleton_vector_store():
    """
    Reset vector store singleton between tests to ensure clean state and isolation.
    """
    from app.api.deps import reset_vector_store
    reset_vector_store()
    yield
    reset_vector_store()
