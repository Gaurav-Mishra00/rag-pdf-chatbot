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
        Extracts raw text content from the PDF file bytes.
        """
        # TODO: Implement text extraction using PdfReader
        # Example:
        # reader = PdfReader(BytesIO(file_content))
        # text = ""
        # for page in reader.pages:
        #     page_text = page.extract_text()
        #     if page_text:
        #         text += page_text + "\n"
        # return text
        return ""

    def process_pdf(self, file_content: bytes, filename: str) -> List[Document]:
        """
        Extracts text and chunks it into list of LangChain Documents.
        """
        # TODO: Extract text and split into documents
        # Example:
        # text = self.extract_text_from_pdf(file_content)
        # raw_docs = [Document(page_content=text, metadata={"source": filename})]
        # return self.splitter.split_documents(raw_docs)
        return []
