from typing import List
from fastapi import APIRouter, Depends, status
from app.api.deps import get_vector_store
from app.core.security import verify_api_key
from app.schemas.document import SearchQuerySchema, SearchResultDocument
from app.vectorstore.faiss_store import FAISSVectorStore

router = APIRouter()


@router.post(
    "/search",
    response_model=List[SearchResultDocument],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def search_vectorstore(
    payload: SearchQuerySchema,
    vector_store: FAISSVectorStore = Depends(get_vector_store),
) -> List[SearchResultDocument]:
    """
    Directly queries the FAISS index to retrieve the most similar text chunks.
    Useful for debugging embeddings retrieval quality without calling the LLM.
    """
    results = vector_store.similarity_search(payload.query, k=payload.top_k)

    # Map output tuples (Document, score) to Pydantic SearchResultDocument
    return [
        SearchResultDocument(
            page_content=doc.page_content,
            metadata=doc.metadata,
            score=float(score),
        )
        for doc, score in results
    ]


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def get_index_status(
    vector_store: FAISSVectorStore = Depends(get_vector_store),
):
    """
    Returns statistics and status of the current FAISS vector store.
    """
    is_active = vector_store.vector_store is not None
    return {
        "status": "ready" if is_active else "uninitialized",
        "index_type": "FAISS",
        "has_local_index": is_active,
    }
