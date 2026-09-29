"""Reindexing CLI script.

Rebuilds the FAISS vector index from documents stored in SQLite and disk using
the active embedding model configured in settings (e.g. BGE-M3).
"""

import os
import sys
import uuid
import logging
from pathlib import Path

from app.core.config import settings
from app.core.database import get_db_connection
from app.services.pdf_processor import PDFProcessorService
from app.vectorstore.faiss_store import FAISSVectorStore
from app.api.deps import _create_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def reindex():
    logger.info("Starting FAISS reindexing with model '%s'...", settings.EMBEDDING_MODEL_NAME)
    index_path = Path(settings.FAISS_INDEX_PATH)
    index_path.mkdir(parents=True, exist_ok=True)

    # 1. Clean up legacy pickle/faiss files in index dir
    for f in index_path.glob("*.pkl"):
        f.unlink(missing_ok=True)
    for f in index_path.glob("*.faiss"):
        f.unlink(missing_ok=True)
    for f in index_path.glob("manifest*.json"):
        f.unlink(missing_ok=True)

    # 2. Fetch completed documents from database
    with get_db_connection() as conn:
        docs = conn.execute(
            "SELECT document_id, filename, file_path, user_id FROM documents WHERE status = 'completed'"
        ).fetchall()

    embeddings = _create_embeddings()
    store = FAISSVectorStore(embeddings=embeddings)

    if not docs:
        logger.info("No documents found in database. Initializing empty index...")
        store.create_empty_index()
        logger.info("Empty index created successfully at '%s'.", index_path)
        return

    processor = PDFProcessorService(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    all_chunks = []
    all_chunk_ids = []

    with get_db_connection() as conn:
        # Clear existing document_chunks mapping before repopulating
        conn.execute("DELETE FROM document_chunks")

        for doc in docs:
            doc_id = doc["document_id"]
            filename = doc["filename"]
            file_path = doc["file_path"]
            user_id = doc["user_id"]

            if not os.path.exists(file_path):
                logger.warning("File '%s' not found on disk, skipping document %s.", file_path, doc_id)
                continue

            with open(file_path, "rb") as f:
                content = f.read()

            chunks = processor.process_pdf(content, filename)
            logger.info("Document '%s' produced %d chunks.", filename, len(chunks))

            chunk_ids = [str(uuid.uuid4()) for _ in chunks]
            for i, chunk in enumerate(chunks):
                chunk.metadata["user_id"] = user_id
                chunk.metadata["document_id"] = doc_id
                chunk.metadata["chunk_id"] = chunk_ids[i]

                conn.execute(
                    "INSERT INTO document_chunks (chunk_id, document_id) VALUES (?, ?)",
                    (chunk_ids[i], doc_id),
                )

            conn.execute(
                "UPDATE documents SET chunk_count = ? WHERE document_id = ?",
                (len(chunks), doc_id),
            )

            all_chunks.extend(chunks)
            all_chunk_ids.extend(chunk_ids)

    if all_chunks:
        logger.info("Indexing %d total chunks into FAISS...", len(all_chunks))
        store.add_documents(all_chunks, ids=all_chunk_ids)
    else:
        store.create_empty_index()

    logger.info("Reindexing complete. Index saved to '%s'.", index_path)


if __name__ == "__main__":
    reindex()
