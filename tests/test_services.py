from unittest.mock import MagicMock, patch, call
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


def test_rag_service_mock_response():
    """
    Test RAG service response orchestration using fake models.
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

    mock_from.assert_called_once_with(SAMPLE_DOCS, fake_store.embeddings)
    mock_save.assert_called_once()
    assert fake_store.vector_store is mock_vs


def test_faiss_add_documents_appends_to_existing(fake_store):
    """add_documents() merges into the existing store without rebuilding."""
    mock_vs = MagicMock()
    mock_vs.index.ntotal = 4
    fake_store.vector_store = mock_vs

    with patch.object(fake_store, "save_index") as mock_save:
        fake_store.add_documents(SAMPLE_DOCS)

    mock_vs.add_documents.assert_called_once_with(SAMPLE_DOCS)
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
