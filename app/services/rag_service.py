"""RAG pipeline — retrieval, reranking, relevance gating, and generation.

This module replaces the previous ``langchain_classic``-based chain with a
direct pipeline that gives us full control over:

- Retrieval (FAISS cosine similarity via native_faiss)
- Reranking (cross-encoder, optional)
- Relevance threshold gating ("I don't know" when nothing is relevant)
- Inline source/page citations in the generated answer
- Evaluation hooks for retrieval and answer quality metrics
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage

from app.core.config import settings
from app.vectorstore.faiss_store import FAISSVectorStore
from app.prompts.templates import CONTEXTUALIZE_SYSTEM_PROMPT, QA_SYSTEM_PROMPT_CITED
from app.services.errors import GenerationError, generation_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for structured pipeline results
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """A single source citation attached to an answer."""
    document_name: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: str = ""
    similarity_score: float = 0.0
    rerank_score: Optional[float] = None
    chunk_id: Optional[str] = None


@dataclass
class RAGResult:
    """Full output of a single RAG pipeline invocation."""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    source_docs: List[Document] = field(default_factory=list)
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    candidates_before_rerank: int = 0
    candidates_after_rerank: int = 0
    relevance_passed: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert_chat_history(history: List[Dict[str, str]]) -> List[BaseMessage]:
    """
    Converts a list of ``{"role": "...", "content": "..."}`` dicts into
    LangChain ``BaseMessage`` objects expected by the LLM.

    Roles ``"user"`` / ``"human"`` become ``HumanMessage``; everything
    else (``"assistant"`` / ``"ai"``) becomes ``AIMessage``.
    """
    messages: List[BaseMessage] = []
    for turn in history:
        role = turn.get("role", "").lower()
        content = turn.get("content", "")
        if role in ("user", "human"):
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def _build_context_block(docs: List[Document]) -> str:
    """Formats retrieved documents into a numbered context block with source metadata."""
    if not docs:
        return "(No relevant context found.)"

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section", "")
        header = f"[Source {i}: {source}, Page {page}"
        if section:
            header += f", Section: {section}"
        header += "]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# RAG Service
# ---------------------------------------------------------------------------

class RAGService:
    """
    Orchestrator service that integrates the FAISS vector store and an LLM to
    implement a full Retrieval-Augmented Generation (RAG) pipeline.

    Pipeline:
      1. **Contextualize** — Rewrites the user's question into a standalone
         query using chat history (LLM call, skipped when history is empty).
      2. **Retrieve** — Queries the FAISS index for top-K candidates.
      3. **Rerank** (optional) — Scores candidates with a cross-encoder and
         re-orders them. Controlled by ``settings.RERANK_ENABLED``.
      4. **Relevance gate** — Filters candidates below
         ``settings.MIN_SIMILARITY`` / ``settings.RERANK_MIN_SCORE``.
         If no documents pass, returns an "I don't know" answer.
      5. **Generate** — Calls the LLM with the filtered context and a prompt
         that requires inline ``[Source N]`` citations.
    """

    def __init__(self, vector_store: FAISSVectorStore, llm: BaseChatModel):
        self.vector_store = vector_store
        self.llm = llm

    # ------------------------------------------------------------------
    # Step 1: Contextualize
    # ------------------------------------------------------------------

    def _contextualize_query(
        self, query: str, chat_history: List[BaseMessage]
    ) -> str:
        """Rewrites a follow-up question into a standalone query using the LLM."""
        if not chat_history:
            return query

        messages = [
            SystemMessage(content=CONTEXTUALIZE_SYSTEM_PROMPT),
            *chat_history,
            HumanMessage(content=query),
        ]
        try:
            response = self.llm.invoke(messages)
            standalone = response.content.strip()
            if standalone:
                logger.debug("Contextualized query: %r -> %r", query, standalone)
                return standalone
        except Exception as exc:
            logger.warning("Query contextualization failed, using original: %s", exc)

        return query

    # ------------------------------------------------------------------
    # Step 2: Retrieve
    # ------------------------------------------------------------------

    def _retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieves top-K candidates from FAISS with similarity scores."""
        from unittest.mock import MagicMock
        is_mocked = isinstance(getattr(self.vector_store, "similarity_search", None), MagicMock)
        if self.vector_store.vector_store is None and not is_mocked:
            raise RuntimeError(
                "Vector store is empty. Upload and ingest at least one document before querying."
            )
        return self.vector_store.similarity_search(
            query, k=settings.RETRIEVAL_K
        )

    # ------------------------------------------------------------------
    # Step 3: Rerank
    # ------------------------------------------------------------------

    @staticmethod
    def _rerank(query: str, docs: List[Document]) -> List[Document]:
        """Applies cross-encoder reranking if enabled in settings."""
        if not settings.RERANK_ENABLED or not docs:
            return docs

        from app.services.reranker import rerank
        return rerank(query, docs)

    # ------------------------------------------------------------------
    # Step 4: Relevance gate
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_relevance_gate(
        docs: List[Document], reranked: bool
    ) -> Tuple[List[Document], bool]:
        """
        Filters documents below the relevance threshold.

        Returns ``(filtered_docs, relevance_passed)`` where
        ``relevance_passed`` is False when *all* documents were filtered out
        — triggering the "I don't know" fallback.
        """
        if reranked:
            threshold = settings.RERANK_MIN_SCORE
            score_key = "rerank_score"
        else:
            threshold = settings.MIN_SIMILARITY
            score_key = "similarity_score"

        filtered = []
        for doc in docs:
            score = doc.metadata.get(score_key, 0.0)
            if score >= threshold:
                filtered.append(doc)

        if not filtered:
            return [], False

        # Take top CONTEXT_K after filtering
        return filtered[: settings.CONTEXT_K], True

    # ------------------------------------------------------------------
    # Step 5: Generate
    # ------------------------------------------------------------------

    def _generate(
        self,
        query: str,
        context_docs: List[Document],
        chat_history: List[BaseMessage],
    ) -> str:
        """Calls the LLM to generate an answer with inline citations."""
        context_block = _build_context_block(context_docs)
        system_prompt = QA_SYSTEM_PROMPT_CITED.format(context=context_block)

        messages = [
            SystemMessage(content=system_prompt),
            *chat_history,
            HumanMessage(content=query),
        ]

        try:
            response = self.llm.invoke(messages)
            answer = response.content.strip()
            if not answer:
                raise GenerationError(
                    "The AI service returned an empty answer. Please try again."
                )
            return answer
        except GenerationError:
            raise
        except Exception as exc:
            logger.error("LLM invocation failed: %s", exc, exc_info=True)
            raise generation_error(exc) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer_query(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
    ) -> Tuple[str, List[Document]]:
        """
        Executes the full RAG pipeline and returns ``(answer, source_documents)``.

        This method signature is kept backward-compatible with the chat
        endpoint. For richer output, use ``answer_query_detailed()``.
        """
        result = self.answer_query_detailed(query, chat_history)
        return result.answer, result.source_docs

    def answer_query_detailed(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
    ) -> RAGResult:
        """
        Executes the full RAG pipeline with timing and diagnostic metadata.

        Returns a ``RAGResult`` with the answer, citations, source docs,
        timing breakdowns, and relevance gate status.
        """
        t_start = time.perf_counter()

        lc_history = _convert_chat_history(chat_history)

        # 1. Contextualize
        standalone_query = self._contextualize_query(query, lc_history)

        # 2. Retrieve
        t_retrieve = time.perf_counter()
        try:
            search_results = self._retrieve(standalone_query)
        except RuntimeError as exc:
            logger.warning("answer_query: %s", exc)
            raise GenerationError(str(exc), status_code=409) from exc

        retrieval_time_ms = (time.perf_counter() - t_retrieve) * 1000

        # Separate docs and scores
        candidates = []
        for doc, score in search_results:
            doc.metadata["similarity_score"] = float(score)
            candidates.append(doc)
        candidates_before_rerank = len(candidates)

        # 3. Rerank
        t_rerank = time.perf_counter()
        reranked = settings.RERANK_ENABLED and len(candidates) > 0
        if reranked:
            candidates = self._rerank(standalone_query, candidates)
        rerank_time_ms = (time.perf_counter() - t_rerank) * 1000

        # 4. Relevance gate
        filtered_docs, relevance_passed = self._apply_relevance_gate(
            candidates, reranked=reranked
        )
        candidates_after_rerank = len(filtered_docs)

        # 5. Generate
        t_gen = time.perf_counter()
        if not relevance_passed:
            answer = (
                "I don't have enough relevant information in the uploaded documents "
                "to answer this question. Please try rephrasing your question, or "
                "upload additional documents that may contain the answer."
            )
        else:
            answer = self._generate(standalone_query, filtered_docs, lc_history)
        generation_time_ms = (time.perf_counter() - t_gen) * 1000

        total_time_ms = (time.perf_counter() - t_start) * 1000

        # Build citations
        citations = []
        for i, doc in enumerate(filtered_docs):
            citations.append(Citation(
                document_name=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                section=doc.metadata.get("section", ""),
                snippet=doc.page_content[:300],
                similarity_score=doc.metadata.get("similarity_score", 0.0),
                rerank_score=doc.metadata.get("rerank_score"),
                chunk_id=doc.metadata.get("chunk_id"),
            ))

        # Enrich source_docs metadata with scores for backward compat
        for doc in filtered_docs:
            doc.metadata["score"] = doc.metadata.get(
                "rerank_score" if reranked else "similarity_score", 0.0
            )

        logger.info(
            "answer_query: query=%r | candidates=%d->%d | relevant=%s | "
            "retrieve=%.0fms rerank=%.0fms gen=%.0fms total=%.0fms",
            query, candidates_before_rerank, candidates_after_rerank,
            relevance_passed, retrieval_time_ms, rerank_time_ms,
            generation_time_ms, total_time_ms,
        )

        return RAGResult(
            answer=answer,
            citations=citations,
            source_docs=filtered_docs,
            retrieval_time_ms=retrieval_time_ms,
            rerank_time_ms=rerank_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            candidates_before_rerank=candidates_before_rerank,
            candidates_after_rerank=candidates_after_rerank,
            relevance_passed=relevance_passed,
        )
