from fastapi import APIRouter, Depends, status
from app.api.deps import get_rag_service
from app.core.security import verify_api_key
from app.schemas.chat import ChatQuery, ChatResponse, SourceDocumentSchema
from app.services.rag_service import RAGService

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
) -> ChatResponse:
    """
    Submits a user prompt/question to the RAG chatbot.
    Retrieves matching documents from FAISS and generates an answer using LLM.
    """
    # Dummy mock structure for answer and sources
    answer, source_docs = rag_service.answer_query(
        query=payload.message, chat_history=[]
    )

    # Convert retrieved documents to API response format
    sources_response = [
        SourceDocumentSchema(
            document_name=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            snippet=doc.page_content[:200],
        )
        for doc in source_docs
    ]

    return ChatResponse(
        session_id=payload.session_id or "default-session",
        answer=answer,
        sources=sources_response,
        metadata={"model_name": "mock-rag-pipeline"},
    )
