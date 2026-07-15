from fastapi import APIRouter
from app.api.endpoints import chat, documents, vectorstore, sessions

api_router = APIRouter()

# Register sub-routes
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(vectorstore.router, prefix="/vectorstore", tags=["Vector Store"])
api_router.include_router(sessions.router, prefix="/sessions", tags=["Sessions"])
