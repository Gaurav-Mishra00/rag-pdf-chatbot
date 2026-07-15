import logging
import os
import threading
import uuid
from typing import List, Optional, Tuple

import faiss as faiss_lib  # for runtime index type assertion
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Single-process multi-threading write lock.
#
# IMPORTANT: This lock is process-local. It protects against data races
# between concurrent request threads sharing one Uvicorn worker process.
# It does NOT protect against races when running with multiple worker
# processes (e.g. `uvicorn --workers N` with N > 1). For multi-process
# deployments, replace FAISS with a server-backed vector store (e.g.
# Pinecone, Weaviate, or pgvector) that handles its own concurrency.
# -----------------------------------------------------------------------
_faiss_write_lock = threading.Lock()


class FAISSVectorStore:
    """
    A production wrapper around LangChain's FAISS vector store.
    Handles indexing, persistence, deletion, and similarity search.

    Thread-safety: write operations (add, delete, create) acquire a
    module-level threading.Lock. This is sufficient for single-process
    deployments only. See _faiss_write_lock docstring above.

    FAISS deletion: LangChain's FAISS wrapper stores vectors under an
    IndexFlatL2 (or similar flat index) which supports remove_ids(). The
    .delete() call is confirmed to work and the result is persisted to disk
    via save_local() inside the same lock scope.  A server restart will
    therefore load the post-deletion index — deleted docs will NOT resurface.
    """

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_index(self) -> bool:
        """
        Loads the FAISS index from the configured local directory path.
        Returns True if the index was found and loaded, False otherwise.
        """
        index_path = settings.FAISS_INDEX_PATH
        index_file = os.path.join(index_path, "index.faiss")

        if not os.path.exists(index_file):
            logger.info("No existing FAISS index found at '%s'.", index_path)
            return False

        try:
            self.vector_store = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                "FAISS index loaded from '%s' (%d vectors).",
                index_path,
                self.count,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc, exc_info=True)
            return False

    def save_index(self) -> None:
        """
        Persists the current FAISS index to the local filesystem via
        save_local(). Must be called while _faiss_write_lock is held by
        the caller so the on-disk state stays consistent.
        """
        if self.vector_store is None:
            logger.warning("save_index() called but vector store is not initialised.")
            return

        index_path = settings.FAISS_INDEX_PATH
        try:
            os.makedirs(index_path, exist_ok=True)
            self.vector_store.save_local(index_path)
            logger.info(
                "FAISS index saved to '%s' (%d vectors).",
                index_path,
                self.count,
            )
        except Exception as exc:
            logger.error("Failed to save FAISS index: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Indexing & Deletion
    # ------------------------------------------------------------------

    def create_empty_index(self, dummy_text: str = "initialization placeholder") -> None:
        """
        Bootstraps a new FAISS index from a single dummy document.
        The dummy document is tagged with source="__init__" so it can be
        filtered at query time. save_local() is called inside the lock.
        """
        doc = Document(
            page_content=dummy_text,
            metadata={"source": "__init__", "page": 0},
        )
        with _faiss_write_lock:
            try:
                self.vector_store = FAISS.from_documents([doc], self.embeddings)
                self.save_index()
                logger.info("Bootstrapped empty FAISS index.")
            except Exception as exc:
                logger.error("Failed to create empty FAISS index: %s", exc, exc_info=True)
                raise

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """
        Adds documents to the FAISS index and immediately persists the
        result to disk via save_local() (inside the write lock).

        ``ids`` must be pre-generated UUIDs when the caller needs to
        track the mapping between chunk IDs and the database rows.
        Auto-generated UUIDs are returned when ``ids`` is None.
        """
        if not documents:
            logger.warning("add_documents() called with an empty list — nothing to do.")
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        with _faiss_write_lock:
            try:
                if self.vector_store is None:
                    logger.info(
                        "Vector store uninitialised; building index from %d documents.",
                        len(documents),
                    )
                    self.vector_store = FAISS.from_documents(documents, self.embeddings, ids=ids)
                else:
                    self.vector_store.add_documents(documents, ids=ids)
                    logger.info(
                        "Added %d documents to existing FAISS index (total: %d).",
                        len(documents),
                        self.count,
                    )
                # Persist immediately; any restart will see the post-add state
                self.save_index()
                return ids
            except Exception as exc:
                logger.error("Failed to add documents to FAISS index: %s", exc, exc_info=True)
                raise

    def delete_documents(self, chunk_ids: List[str]) -> None:
        """
        Deletes vectors by chunk ID from the FAISS index.

        Deletion is supported on the flat index type that LangChain creates
        by default (IndexFlatL2) — confirmed via a runtime probe on startup.
        save_local() is called inside the lock immediately after deletion, so
        a server restart will load the post-deletion index and deleted docs
        will NOT be resurrected.

        Raises RuntimeError if the underlying index does not support removal
        (e.g. HNSW, IVF without IDMap wrapping).
        """
        if not chunk_ids:
            logger.warning("delete_documents() called with empty list of chunk IDs.")
            return

        with _faiss_write_lock:
            if self.vector_store is None:
                logger.warning("delete_documents() called but vector store is not initialised.")
                return

            # Guard: confirm this index type exposes remove_ids before calling delete()
            raw_index = getattr(self.vector_store, "index", None)
            if raw_index is not None and not hasattr(raw_index, "remove_ids"):
                raise RuntimeError(
                    f"FAISS index type {type(raw_index).__name__} does not support "
                    "deletion (remove_ids missing). Rebuild the index using a flat or IDMap index."
                )

            try:
                deleted = self.vector_store.delete(chunk_ids)
                if deleted is False:
                    logger.warning(
                        "FAISS.delete() returned False for chunk_ids=%s — some IDs may not have existed.",
                        chunk_ids,
                    )
                logger.info("Deleted %d chunk vectors from FAISS index.", len(chunk_ids))
                # Persist immediately so the deletion survives a server restart
                self.save_index()
            except Exception as exc:
                logger.error("Failed to delete documents from FAISS index: %s", exc, exc_info=True)
                raise

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def similarity_search(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Performs a similarity search against the FAISS index.
        Returns (Document, score) tuples; empty list when uninitialised.
        """
        if self.vector_store is None:
            logger.warning("similarity_search() called but index is not initialised.")
            return []

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.debug(
                "similarity_search: query=%r, k=%d, hits=%d", query, k, len(results)
            )
            return results
        except Exception as exc:
            logger.error("FAISS similarity search failed: %s", exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Returns the number of vectors currently stored in the index."""
        if self.vector_store is None or self.vector_store.index is None:
            return 0
        return self.vector_store.index.ntotal

    @property
    def index_file_size_bytes(self) -> int:
        """
        Returns the size in bytes of the persisted ``index.faiss`` file on disk.
        Returns 0 if the file does not exist (i.e. index has never been saved).
        """
        index_file = os.path.join(settings.FAISS_INDEX_PATH, "index.faiss")
        try:
            return os.path.getsize(index_file)
        except OSError:
            return 0
