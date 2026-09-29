from typing import Generator, Optional
import threading
from fastapi import Depends, HTTPException
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# Fake providers are used only in the testing environment.
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

# For actual implementations:
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from app.core.config import settings
from app.vectorstore.faiss_store import FAISSVectorStore
from app.services.pdf_processor import PDFProcessorService
from app.services.rag_service import RAGService
from app.services.history_manager import HistoryManager
from app.vectorstore.native_faiss import IndexCompatibilityError

# Singleton instances
_history_manager = HistoryManager()
_embeddings: Optional[Embeddings] = None
_embeddings_lock = threading.Lock()
_vector_store: Optional[FAISSVectorStore] = None
_vector_store_lock = threading.Lock()
_rag_service: Optional[RAGService] = None
_rag_service_lock = threading.Lock()


def get_history_manager() -> HistoryManager:
    """
    FastAPI dependency that returns the HistoryManager singleton instance.
    """
    return _history_manager


def get_embeddings() -> Embeddings:
    """Reuse one embedding model per process, including concurrent first requests.

    Restart the process after changing provider/model configuration. An existing
    FAISS index must always be queried with the embedding model that created it.
    """
    global _embeddings
    with _embeddings_lock:
        if _embeddings is None:
            _embeddings = _create_embeddings()
        return _embeddings


def _create_embeddings() -> Embeddings:
    """
    FastAPI dependency that returns the configured Embeddings provider.
    """
    if settings.EMBEDDINGS_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL_NAME,
        )
    elif settings.EMBEDDINGS_PROVIDER == "google" and settings.GOOGLE_API_KEY:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            google_api_key=settings.GOOGLE_API_KEY,
            model=settings.EMBEDDING_MODEL_NAME,
        )
    elif settings.EMBEDDINGS_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True, "batch_size": settings.EMBEDDING_BATCH_SIZE},
        )
    if settings.APP_ENV == "testing":
        return FakeEmbeddings(size=1536)
    raise HTTPException(503, "Embeddings provider API key is missing. Configure the server's .env file.")


def get_llm() -> BaseChatModel:
    """
    FastAPI dependency that returns the configured Chat LLM provider.
    """
    if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL_NAME,
            temperature=settings.TEMPERATURE,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=1,
        )
    elif settings.LLM_PROVIDER == "google" and settings.GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=settings.GOOGLE_API_KEY,
            model=settings.LLM_MODEL_NAME,
            temperature=settings.TEMPERATURE,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            # This adapter counts total attempts: two means one retry.
            max_retries=2,
        )
    elif settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.LLM_MODEL_NAME,
            temperature=settings.TEMPERATURE,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=1,
        )
    if settings.APP_ENV == "testing":
        return FakeListChatModel(responses=["This is a mock response from FakeListChatModel."])
    raise HTTPException(503, "AI provider API key is missing. Configure the server's .env file.")


def get_vector_store(embeddings: Embeddings = Depends(get_embeddings)) -> FAISSVectorStore:
    """
    FastAPI dependency that returns the shared FAISSVectorStore singleton instance.
    Initializes the store once (thread-safe) on first use.
    """
    global _vector_store
    
    # Resolve FastAPI Depends default argument if called as a normal function in unit tests
    if type(embeddings).__name__ == "Depends":
        embeddings = get_embeddings()

    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                store = FAISSVectorStore(embeddings=embeddings)
                # Attempts to load existing FAISS index on disk
                try:
                    loaded = store.load_index()
                except (IndexCompatibilityError, ValueError, OSError) as exc:
                    raise HTTPException(503, "Vector index requires migration or recovery. Run python -m scripts.reindex with the server stopped.") from exc
                if not loaded:
                    # Create an empty template index if not found
                    store.create_empty_index()
                _vector_store = store
    return _vector_store


def reset_vector_store() -> None:
    """
    Resets the vector store singleton instance. Useful for testing isolation.
    """
    global _vector_store, _rag_service, _embeddings
    with _vector_store_lock:
        _vector_store = None
    with _rag_service_lock:
        _rag_service = None
    with _embeddings_lock:
        _embeddings = None


def get_pdf_processor() -> PDFProcessorService:
    """
    FastAPI dependency that returns the PDF Processor service.
    """
    return PDFProcessorService(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)


def get_rag_service(
    vector_store: FAISSVectorStore = Depends(get_vector_store),
    llm: BaseChatModel = Depends(get_llm),
) -> RAGService:
    """
    FastAPI dependency that returns the shared RAGService singleton instance.
    Initializes the service once (thread-safe) on first use.
    """
    global _rag_service
    if _rag_service is None:
        with _rag_service_lock:
            if _rag_service is None:
                _rag_service = RAGService(vector_store=vector_store, llm=llm)
    return _rag_service
