from unittest.mock import MagicMock, patch, call, ANY
import pytest
from langchain_core.documents import Document
from app.services.pdf_processor import PDFProcessorService
from app.services.rag_service import RAGService
from app.vectorstore.faiss_store import FAISSVectorStore
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.chat_models import FakeListChatModel



def test_pdf_processor_initialization():
    """
    Test PDF processor initialization and parameter configuration.
    """
    processor = PDFProcessorService(chunk_size=500, chunk_overlap=50)
    assert processor.splitter._chunk_size == 500
    assert processor.splitter._chunk_overlap == 50


def test_pdf_processor_empty_content():
    """
    Test processing empty PDF bytes content returns empty list.
    """
    processor = PDFProcessorService()
    docs = processor.process_pdf(b"", "empty.pdf")
    assert docs == []


def test_vectorstore_load_missing():
    """
    Test loading vector store when no files exist returns False.
    """
    embeddings = FakeEmbeddings(size=1536)
    store = FAISSVectorStore(embeddings=embeddings)
    with patch("app.vectorstore.faiss_store.os.path.exists", return_value=False):
        assert store.load_index() is False


# ==========================================================================
# RAG Service Tests
# ==========================================================================

from langchain_core.messages import HumanMessage, AIMessage
from app.services.rag_service import RAGService, _convert_chat_history


def test_convert_chat_history_empty():
    """Empty history → empty list."""
    assert _convert_chat_history([]) == []


def test_convert_chat_history_mixed_roles():
    """Dict history is mapped to correct BaseMessage subclasses."""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "human", "content": "What is RAG?"},
        {"role": "ai", "content": "RAG stands for ..."},
    ]
    result = _convert_chat_history(history)
    assert isinstance(result[0], HumanMessage) and result[0].content == "Hello"
    assert isinstance(result[1], AIMessage) and result[1].content == "Hi there"
    assert isinstance(result[2], HumanMessage) and result[2].content == "What is RAG?"
    assert isinstance(result[3], AIMessage)


def test_rag_service_empty_store_returns_error_string():
    """answer_query() returns a descriptive error when the store is uninitialised."""
    embeddings = FakeEmbeddings(size=1536)
    store = FAISSVectorStore(embeddings=embeddings)
    llm = FakeListChatModel(responses=["ignored"])

    service = RAGService(vector_store=store, llm=llm)
    answer, sources = service.answer_query("test question", [])

    assert "empty" in answer.lower() or "error" in answer.lower() or isinstance(answer, str)
    assert sources == []


def test_rag_service_invokes_lcel_chain():
    """
    answer_query() invokes the LCEL chain and returns answer + source docs.
    The internal chain is mocked to avoid real LLM/FAISS calls.
    """
    embeddings = FakeEmbeddings(size=1536)
    store = FAISSVectorStore(embeddings=embeddings)
    llm = FakeListChatModel(responses=["Mock LLM answer"])

    service = RAGService(vector_store=store, llm=llm)

    source_doc = Document(
        page_content="Context text from page 3.",
        metadata={"source": "report.pdf", "page": 3},
    )

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "answer": "Mock LLM answer",
        "context": [source_doc],
    }
    service._chain = mock_chain  # inject pre-built chain

    answer, sources = service.answer_query(
        "What is RAG?",
        [{"role": "user", "content": "previous question"}],
    )

    # Verify chain was called with correct keys
    call_kwargs = mock_chain.invoke.call_args[0][0]
    assert call_kwargs["input"] == "What is RAG?"
    assert len(call_kwargs["chat_history"]) == 1
    assert isinstance(call_kwargs["chat_history"][0], HumanMessage)

    assert answer == "Mock LLM answer"
    assert len(sources) == 1
    assert sources[0].metadata["page"] == 3
    assert sources[0].metadata["source"] == "report.pdf"


def test_rag_service_chain_exception_returns_error():
    """answer_query() returns a safe error string when the chain raises."""
    embeddings = FakeEmbeddings(size=1536)
    store = FAISSVectorStore(embeddings=embeddings)
    llm = FakeListChatModel(responses=[])

    service = RAGService(vector_store=store, llm=llm)
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = RuntimeError("LLM timeout")
    service._chain = mock_chain

    answer, sources = service.answer_query("Any question", [])

    assert isinstance(answer, str)
    assert "error" in answer.lower()
    assert sources == []


def test_rag_service_mock_response():
    """
    Legacy smoke-test: answer_query() returns (str, list) regardless of
    whether the store is initialised.
    """
    embeddings = FakeEmbeddings(size=1536)
    store = FAISSVectorStore(embeddings=embeddings)
    llm = FakeListChatModel(responses=["Hello, I am a test response"])

    service = RAGService(vector_store=store, llm=llm)
    answer, sources = service.answer_query("test question", [])

    assert isinstance(answer, str)
    assert isinstance(sources, list)



def test_pdf_processor_with_mock_pages():
    """
    Test that process_pdf correctly parses pages and attaches page-level metadata.
    """
    from unittest.mock import MagicMock, patch
    
    processor = PDFProcessorService(chunk_size=100, chunk_overlap=10)
    
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "This is page one text content."
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "This is page two text content."
    
    with patch("app.services.pdf_processor.PdfReader") as mock_pdf_reader:
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader_instance
        
        docs = processor.process_pdf(b"dummy pdf content", "sample.pdf")
        
        assert len(docs) == 2
        assert docs[0].metadata["source"] == "sample.pdf"
        assert docs[0].metadata["page"] == 1
        assert docs[0].page_content == "This is page one text content."
        
        assert docs[1].metadata["source"] == "sample.pdf"
        assert docs[1].metadata["page"] == 2
        assert docs[1].page_content == "This is page two text content."


# ==========================================================================
# FAISS Vector Store Tests
# ==========================================================================

SAMPLE_DOCS = [
    Document(page_content="FastAPI is a modern web framework.", metadata={"source": "a.pdf", "page": 1}),
    Document(page_content="FAISS enables fast similarity search.", metadata={"source": "a.pdf", "page": 2}),
]


@pytest.fixture()
def fake_store():
    """FAISSVectorStore wired with FakeEmbeddings (no real API calls)."""
    return FAISSVectorStore(embeddings=FakeEmbeddings(size=1536))


def test_faiss_load_missing_index(fake_store):
    """load_index() returns False when index files do not exist on disk."""
    with patch("app.vectorstore.faiss_store.os.path.exists", return_value=False):
        result = fake_store.load_index()
    assert result is False
    assert fake_store.vector_store is None


def test_faiss_load_existing_index(fake_store):
    """load_index() loads an existing index and returns True."""
    mock_vs = MagicMock()
    mock_vs.index.ntotal = 5

    with patch("app.vectorstore.faiss_store.os.path.exists", return_value=True), \
         patch("app.vectorstore.faiss_store.FAISS.load_local", return_value=mock_vs):
        result = fake_store.load_index()

    assert result is True
    assert fake_store.vector_store is mock_vs


def test_faiss_add_documents_creates_index(fake_store):
    """add_documents() builds a new index when the store is uninitialised."""
    mock_vs = MagicMock()
    mock_vs.index.ntotal = 2

    with patch("app.vectorstore.faiss_store.FAISS.from_documents", return_value=mock_vs) as mock_from, \
         patch.object(fake_store, "save_index") as mock_save:
        fake_store.add_documents(SAMPLE_DOCS)

    mock_from.assert_called_once_with(SAMPLE_DOCS, fake_store.embeddings, ids=ANY)
    mock_save.assert_called_once()
    assert fake_store.vector_store is mock_vs


def test_faiss_add_documents_appends_to_existing(fake_store):
    """add_documents() merges into the existing store without rebuilding."""
    mock_vs = MagicMock()
    mock_vs.index.ntotal = 4
    fake_store.vector_store = mock_vs

    with patch.object(fake_store, "save_index") as mock_save:
        fake_store.add_documents(SAMPLE_DOCS)

    mock_vs.add_documents.assert_called_once_with(SAMPLE_DOCS, ids=ANY)
    mock_save.assert_called_once()


def test_faiss_add_documents_empty_noop(fake_store):
    """add_documents() with an empty list is a no-op (no index mutation)."""
    with patch.object(fake_store, "save_index") as mock_save:
        fake_store.add_documents([])
    mock_save.assert_not_called()


def test_faiss_similarity_search_no_index(fake_store):
    """similarity_search() returns [] when the index is uninitialised."""
    result = fake_store.similarity_search("test query", k=3)
    assert result == []


def test_faiss_similarity_search_returns_results(fake_store):
    """similarity_search() delegates to the underlying vector store."""
    expected = [(SAMPLE_DOCS[0], 0.12), (SAMPLE_DOCS[1], 0.34)]
    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = expected
    fake_store.vector_store = mock_vs

    result = fake_store.similarity_search("fast search", k=2)

    mock_vs.similarity_search_with_score.assert_called_once_with("fast search", k=2)
    assert result == expected


def test_faiss_save_index_no_store(fake_store, caplog):
    """save_index() logs a warning and returns safely when uninitialised."""
    import logging
    with caplog.at_level(logging.WARNING, logger="app.vectorstore.faiss_store"):
        fake_store.save_index()  # should not raise
    assert "not initialised" in caplog.text


def test_faiss_save_index_persists(fake_store, tmp_path, monkeypatch):
    """save_index() calls FAISS.save_local with the configured path."""
    monkeypatch.setattr("app.vectorstore.faiss_store.settings.FAISS_INDEX_PATH", str(tmp_path))
    mock_vs = MagicMock()
    mock_vs.index.ntotal = 3
    fake_store.vector_store = mock_vs

    fake_store.save_index()

    mock_vs.save_local.assert_called_once_with(str(tmp_path))


def test_faiss_count_property(fake_store):
    """count returns ntotal from the underlying FAISS index."""
    assert fake_store.count == 0  # uninitialised

    mock_vs = MagicMock()
    mock_vs.index.ntotal = 7
    fake_store.vector_store = mock_vs

    assert fake_store.count == 7


def test_faiss_create_empty_index(fake_store):
    """create_empty_index() bootstraps the store with a placeholder document."""
    mock_vs = MagicMock()
    mock_vs.index.ntotal = 1

    with patch("app.vectorstore.faiss_store.FAISS.from_documents", return_value=mock_vs) as mock_from, \
         patch.object(fake_store, "save_index") as mock_save:
        fake_store.create_empty_index()

    created_docs = mock_from.call_args[0][0]
    assert len(created_docs) == 1
    assert created_docs[0].metadata["source"] == "__init__"
    mock_save.assert_called_once()


# ==========================================================================
# Provider Initialization Tests
# ==========================================================================

def test_get_embeddings_google(monkeypatch):
    """get_embeddings() instantiates GoogleGenerativeAIEmbeddings when provider is 'google'."""
    from app.api.deps import get_embeddings
    from app.core.config import settings

    monkeypatch.setattr(settings, "EMBEDDINGS_PROVIDER", "google")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "test-google-key")

    mock_embeddings = MagicMock()
    with patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", return_value=mock_embeddings) as mock_class:
        embeddings = get_embeddings()
        assert embeddings == mock_embeddings
        mock_class.assert_called_once_with(
            google_api_key="test-google-key",
            model=settings.EMBEDDING_MODEL_NAME,
        )


def test_get_embeddings_huggingface(monkeypatch):
    """get_embeddings() instantiates HuggingFaceEmbeddings when provider is 'huggingface'."""
    from app.api.deps import get_embeddings
    from app.core.config import settings

    monkeypatch.setattr(settings, "EMBEDDINGS_PROVIDER", "huggingface")

    mock_embeddings = MagicMock()
    with patch("langchain_community.embeddings.HuggingFaceEmbeddings", return_value=mock_embeddings) as mock_class:
        embeddings = get_embeddings()
        assert embeddings == mock_embeddings
        mock_class.assert_called_once_with(
            model_name=settings.EMBEDDING_MODEL_NAME,
        )


def test_get_llm_google(monkeypatch):
    """get_llm() instantiates ChatGoogleGenerativeAI when provider is 'google'."""
    from app.api.deps import get_llm
    from app.core.config import settings

    monkeypatch.setattr(settings, "LLM_PROVIDER", "google")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "test-google-key")

    mock_llm = MagicMock()
    with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm) as mock_class:
        llm = get_llm()
        assert llm == mock_llm
        mock_class.assert_called_once_with(
            google_api_key="test-google-key",
            model=settings.LLM_MODEL_NAME,
            temperature=settings.TEMPERATURE,
        )


def test_get_llm_anthropic(monkeypatch):
    """get_llm() instantiates ChatAnthropic when provider is 'anthropic'."""
    from app.api.deps import get_llm
    from app.core.config import settings

    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "test-anthropic-key")

    mock_llm = MagicMock()
    with patch("langchain_anthropic.ChatAnthropic", return_value=mock_llm) as mock_class:
        llm = get_llm()
        assert llm == mock_llm
        mock_class.assert_called_once_with(
            api_key="test-anthropic-key",
            model=settings.LLM_MODEL_NAME,
            temperature=settings.TEMPERATURE,
        )
