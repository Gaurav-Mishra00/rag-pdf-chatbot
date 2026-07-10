from typing import List
from io import BytesIO
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class PDFProcessorService:
    """
    Service to process uploaded PDF files, extract text content, and split it
    into chunks suitable for vector embedding and retrieval.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """
        Extracts raw text content from the PDF file bytes across all pages.
        """
        if not file_content:
            return ""
        try:
            reader = PdfReader(BytesIO(file_content))
            text = ""
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception:
                    continue
            return text
        except Exception:
            return ""

    def process_pdf(self, file_content: bytes, filename: str) -> List[Document]:
        """
        Parses a PDF file from bytes, extracts text from each page, 
        creates a Document for each page with metadata containing the 
        source filename and the 1-indexed page number, and splits them 
        into smaller text chunks.
        """
        if not file_content:
            return []

        try:
            reader = PdfReader(BytesIO(file_content))
        except Exception:
            return []

        page_documents = []
        for page_idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
            except Exception:
                page_text = ""
            
            # Skip empty pages or pages with no extractable text
            if page_text and page_text.strip():
                metadata = {
                    "source": filename,
                    "page": page_idx + 1,
                }
                page_documents.append(Document(page_content=page_text, metadata=metadata))

        if not page_documents:
            return []

        return self.splitter.split_documents(page_documents)
