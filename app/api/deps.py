from typing import Generator
from fastapi import Depends
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# We import mock/fake objects if keys are not present to allow application startup
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.chat_models import FakeListChatModel

# For actual implementations:
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from app.core.config import settings
from app.vectorstore.faiss_store import FAISSVectorStore
from app.services.pdf_processor import PDFProcessorService
from app.services.rag_service import RAGService
from app.services.history_manager import HistoryManager

# Singleton instance for in-memory store
_history_manager = HistoryManager()


def get_history_manager() -> HistoryManager:
    """
    FastAPI dependency that returns the HistoryManager singleton instance.
    """
    return _history_manager


def get_embeddings() -> Embeddings:
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
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME
        )
    # Fallback to Fake/Mock Embeddings if not configured or keys are missing
    return FakeEmbeddings(size=1536)


def get_llm() -> BaseChatModel:
    """
    FastAPI dependency that returns the configured Chat LLM provider.
    """
    if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL_NAME,
            temperature=settings.TEMPERATURE,
        )
    elif settings.LLM_PROVIDER == "google" and settings.GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=settings.GOOGLE_API_KEY,
            model=settings.LLM_MODEL_NAME,
            temperature=settings.TEMPERATURE,
        )
    elif settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.LLM_MODEL_NAME,
            temperature=settings.TEMPERATURE,
        )
    # Fallback to Fake/Mock LLM if not configured
    return FakeListChatModel(responses=["This is a mock response from FakeListChatModel."])


def get_vector_store(embeddings: Embeddings = Depends(get_embeddings)) -> FAISSVectorStore:
    """
    FastAPI dependency that returns the FAISSVectorStore instance.
    """
    store = FAISSVectorStore(embeddings=embeddings)
    # Attempts to load existing FAISS index on disk
    loaded = store.load_index()
    if not loaded:
        # Create an empty template index if not found
        store.create_empty_index()
    return store


def get_pdf_processor() -> PDFProcessorService:
    """
    FastAPI dependency that returns the PDF Processor service.
    """
    return PDFProcessorService()


def get_rag_service(
    vector_store: FAISSVectorStore = Depends(get_vector_store),
    llm: BaseChatModel = Depends(get_llm),
) -> RAGService:
    """
    FastAPI dependency that returns the RAGService orchestrator.
    """
    return RAGService(vector_store=vector_store, llm=llm)
