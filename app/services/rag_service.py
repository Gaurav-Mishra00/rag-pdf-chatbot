import logging
from typing import Dict, List, Tuple

from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from app.vectorstore.faiss_store import FAISSVectorStore
from app.prompts.templates import contextualize_prompt, qa_prompt

logger = logging.getLogger(__name__)


def _convert_chat_history(history: List[Dict[str, str]]) -> List[BaseMessage]:
    """
    Converts a list of ``{"role": "...", "content": "..."}`` dicts into
    LangChain ``BaseMessage`` objects expected by the LCEL chains.

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


class RAGService:
    """
    Orchestrator service that integrates the FAISS vector store and an LLM to
    implement a full Retrieval-Augmented Generation (RAG) pipeline using
    LangChain Expression Language (LCEL).

    Pipeline:
      1. ``create_history_aware_retriever`` — rewrites the user's question
         into a self-contained query using ``contextualize_prompt`` and
         chat history before hitting the vector store.
      2. ``create_stuff_documents_chain`` — formats retrieved docs with
         ``qa_prompt`` and calls the LLM to produce the answer.
      3. ``create_retrieval_chain`` — wires (1) and (2) into a single
         runnable that accepts ``{input, chat_history}`` and returns
         ``{answer, context}``.
    """

    def __init__(self, vector_store: FAISSVectorStore, llm: BaseChatModel):
        self.vector_store = vector_store
        self.llm = llm
        self._chain = None  # built lazily on first call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chain(self):
        """
        Constructs and caches the LCEL retrieval chain.
        Called lazily so the chain is only built when the vector store
        has been populated (non-None ``self.vector_store.vector_store``).
        """
        if self.vector_store.vector_store is None:
            raise RuntimeError(
                "Vector store is empty. Upload and ingest at least one document before querying."
            )

        retriever = self.vector_store.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        # Step 1 — history-aware retriever: rewrites question using history
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_prompt
        )

        # Step 2 — document QA chain: stuffs retrieved docs into qa_prompt
        qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # Step 3 — full retrieval chain
        self._chain = create_retrieval_chain(history_aware_retriever, qa_chain)
        logger.debug("RAG chain built successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer_query(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
    ) -> Tuple[str, List[Document]]:
        """
        Executes the full RAG pipeline:

        1. Converts ``chat_history`` dicts → LangChain ``BaseMessage`` objects.
        2. Runs the LCEL chain with ``{"input": query, "chat_history": messages}``.
        3. Returns ``(answer_string, source_documents)``.

        ``source_documents`` preserve the original ``metadata`` dict
        (including ``source`` and ``page``) attached during PDF ingestion.
        """
        if self._chain is None:
            try:
                self._build_chain()
            except RuntimeError as exc:
                logger.warning("answer_query: %s", exc)
                return str(exc), []

        lc_history = _convert_chat_history(chat_history)

        try:
            result = self._chain.invoke(
                {"input": query, "chat_history": lc_history}
            )
        except Exception as exc:
            logger.error("RAG chain invocation failed: %s", exc, exc_info=True)
            return "An error occurred while generating the response.", []

        answer: str = result.get("answer", "")
        source_docs: List[Document] = result.get("context", [])

        # Fetch similarity search scores to enrich metadata
        scores_map = {}
        try:
            search_results = self.vector_store.similarity_search(query, k=8)
            for res_doc, score in search_results:
                scores_map[res_doc.page_content] = float(score)
        except Exception as exc:
            logger.warning("Could not retrieve similarity scores: %s", exc)

        # Enrich source_docs with the calculated scores
        for doc in source_docs:
            doc.metadata["score"] = scores_map.get(doc.page_content, 0.0)

        logger.info(
            "answer_query: query=%r | sources=%d", query, len(source_docs)
        )
        return answer, source_docs
