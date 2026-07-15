import os
from typing import List

from fastapi import APIRouter, Depends, status

from app.api.deps import get_vector_store
from app.core.config import settings
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
    Useful for debugging embedding retrieval quality without calling the LLM.
    """
    results = vector_store.similarity_search(payload.query, k=payload.top_k)
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
) -> dict:
    """
    Returns statistics and health status of the FAISS vector store index.
    Includes vector count, on-disk file size, and index path.
    """
    is_active = vector_store.vector_store is not None
    file_size_bytes = vector_store.index_file_size_bytes
    return {
        "status": "ready" if is_active else "uninitialized",
        "index_type": "FAISS",
        "has_local_index": is_active,
        "vector_count": vector_store.count,
        "index_path": os.path.basename(settings.FAISS_INDEX_PATH),
        "index_file_size_kb": round(file_size_bytes / 1024, 2) if file_size_bytes else 0,
    }


@router.get(
    "/count",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def get_vector_count(
    vector_store: FAISSVectorStore = Depends(get_vector_store),
) -> dict:
    """
    Lightweight endpoint that returns just the current vector count.
    Useful for polling and monitoring without loading full status.
    """
    return {"count": vector_store.count}
