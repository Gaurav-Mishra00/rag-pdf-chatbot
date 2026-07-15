from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: MessageRole
    content: str


class ChatQuery(BaseModel):
    message: str = Field(..., max_length=4000, description="The user prompt to process")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID")


class SourceDocumentSchema(BaseModel):
    document_name: str
    page: Optional[int] = None
    snippet: str
    score: Optional[float] = None


class ChatResponseMetadata(BaseModel):
    """Typed metadata returned with every chat response."""
    model_name: str
    llm_provider: str
    embeddings_provider: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceDocumentSchema] = Field(default_factory=list)
    metadata: ChatResponseMetadata


# ---------------------------------------------------------------------------
# Session schemas
# ---------------------------------------------------------------------------

class SessionSummary(BaseModel):
    """Summary row for a single chat session (used in list response)."""
    session_id: str
    message_count: int
    last_activity: Optional[datetime] = None


class SessionListResponse(BaseModel):
    """Response model for GET /sessions."""
    sessions: List[SessionSummary]
    total: int


class SessionHistoryResponse(BaseModel):
    """Response model for GET /sessions/{session_id}."""
    session_id: str
    messages: List[Message]
    total: int
