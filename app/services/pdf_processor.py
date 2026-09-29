from typing import List
from io import BytesIO
from collections import Counter
import hashlib
import re
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
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
            add_start_index=True,
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

        # Keep page boundaries so every chunk has an unambiguous citation.
        extracted_pages = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = re.sub(r"(\w)-\n(?=\w)", r"\1", text)
            text = re.sub(r"[^\S\n]+", " ", text.replace("\r\n", "\n"))
            extracted_pages.append(text.strip())
        boundaries = Counter()
        for text in extracted_pages:
            lines = text.splitlines()
            boundaries.update(set(lines[:1] + lines[-1:]))
        repeated = {line for line, count in boundaries.items()
                    if len(extracted_pages) >= 3 and count >= max(3, len(extracted_pages) * 0.6)}
        content_hash = hashlib.sha256(file_content).hexdigest()
        page_documents = []
        active_section = ""
        for page_idx, page in enumerate(reader.pages):
            lines = extracted_pages[page_idx].splitlines()
            if lines and lines[0] in repeated:
                lines = lines[1:]
            if lines and lines[-1] in repeated:
                lines = lines[:-1]
            page_text = "\n".join(lines).strip()
            first_line = lines[0].strip() if lines else ""
            if first_line and len(first_line) <= 100 and (
                first_line.isupper() or re.match(r"^(?:\d+(?:\.\d+)*\s+|Chapter\s+|Section\s+)", first_line, re.I)
            ):
                active_section = first_line
            
            # Skip empty pages or pages with no extractable text
            if page_text and page_text.strip():
                metadata = {
                    "source": filename,
                    "page": page_idx + 1,
                    "page_start": page_idx + 1,
                    "page_end": page_idx + 1,
                    "page_count": len(extracted_pages),
                    "section": active_section,
                    "content_hash": content_hash,
                    "chunker_version": "page-paragraph-v2",
                }
                page_documents.append(Document(page_content=page_text, metadata=metadata))

        if not page_documents:
            return []

        chunks = self.splitter.split_documents(page_documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": index,
                "char_count": len(chunk.page_content),
                "word_count": len(chunk.page_content.split()),
                "end_index": chunk.metadata["start_index"] + len(chunk.page_content),
                "chunk_hash": hashlib.sha256(chunk.page_content.encode()).hexdigest(),
            })

        return self._filter_duplicate_chunks(chunks)

    def _filter_duplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        from app.core.config import settings
        if not getattr(settings, "CHUNK_DEDUPLICATION_ENABLED", True):
            return chunks
        seen_hashes = set()
        deduped = []
        for chunk in chunks:
            chash = chunk.metadata.get("chunk_hash") or hashlib.sha256(chunk.page_content.encode()).hexdigest()
            if chash not in seen_hashes:
                seen_hashes.add(chash)
                deduped.append(chunk)
        return deduped
