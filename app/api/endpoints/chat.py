import uuid
from fastapi import APIRouter, Depends, status
from app.api.deps import get_rag_service, get_history_manager
from app.core.config import settings
from app.core.security import verify_api_key
from app.schemas.chat import ChatQuery, ChatResponse, ChatResponseMetadata, SourceDocumentSchema
from app.services.rag_service import RAGService
from app.services.history_manager import HistoryManager

router = APIRouter()


@router.post(
    "/query",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def chat_query(
    payload: ChatQuery,
    rag_service: RAGService = Depends(get_rag_service),
    history_manager: HistoryManager = Depends(get_history_manager),
) -> ChatResponse:
    """
    Submits a user prompt/question to the RAG chatbot.
    Retrieves matching documents from FAISS and generates an answer using LLM.
    Persists and updates conversational history across sessions.
    """
    session_id = payload.session_id or f"session-{uuid.uuid4()}"

    # 1. Retrieve history
    history = await history_manager.get_history(session_id)

    # 2. Run query using history
    answer, source_docs = rag_service.answer_query(
        query=payload.message, chat_history=history
    )

    # 3. Save turn to history
    await history_manager.add_message(session_id, "user", payload.message)
    await history_manager.add_message(session_id, "assistant", answer)

    # 4. Map sources with scores
    sources_response = [
        SourceDocumentSchema(
            document_name=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            snippet=doc.page_content,
            score=doc.metadata.get("score"),
        )
        for doc in source_docs
    ]

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=sources_response,
        metadata=ChatResponseMetadata(
            model_name=settings.LLM_MODEL_NAME,
            llm_provider=settings.LLM_PROVIDER,
            embeddings_provider=settings.EMBEDDINGS_PROVIDER,
        ),
    )
