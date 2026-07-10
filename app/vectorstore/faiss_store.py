import logging
import os
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.config import settings

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    A production wrapper around LangChain's FAISS vector store.
    Handles indexing, persistence, and similarity search operations.
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
        Persists the current FAISS index to the local filesystem.
        Creates parent directories automatically.
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
    # Indexing
    # ------------------------------------------------------------------

    def create_empty_index(self, dummy_text: str = "initialization placeholder") -> None:
        """
        Bootstraps a new FAISS index from a single dummy document so the
        underlying index structure exists before real documents are added.
        The dummy document is tagged with ``source="__init__"`` so it can
        be identified and filtered out at query time if needed.
        """
        doc = Document(
            page_content=dummy_text,
            metadata={"source": "__init__", "page": 0},
        )
        try:
            self.vector_store = FAISS.from_documents([doc], self.embeddings)
            self.save_index()
            logger.info("Bootstrapped empty FAISS index.")
        except Exception as exc:
            logger.error("Failed to create empty FAISS index: %s", exc, exc_info=True)
            raise

    def add_documents(self, documents: List[Document]) -> None:
        """
        Adds a list of LangChain Documents to the FAISS index and
        immediately persists the updated index to disk.

        If the index has not been initialised yet, it is created from
        the supplied documents first.
        """
        if not documents:
            logger.warning("add_documents() called with an empty list — nothing to do.")
            return

        try:
            if self.vector_store is None:
                logger.info(
                    "Vector store uninitialised; building index from %d documents.",
                    len(documents),
                )
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)
                logger.info(
                    "Added %d documents to existing FAISS index (total: %d).",
                    len(documents),
                    self.count,
                )
            self.save_index()
        except Exception as exc:
            logger.error("Failed to add documents to FAISS index: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def similarity_search(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Performs a similarity search against the FAISS index.

        Returns a list of ``(Document, score)`` tuples ordered by
        relevance (lowest L2 distance / highest cosine similarity first).
        Returns an empty list when the index is uninitialised or empty.
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
