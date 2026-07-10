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
