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
    message: str = Field(..., description="The user prompt to process")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID")


class SourceDocumentSchema(BaseModel):
    document_name: str
    page: Optional[int] = None
    snippet: str
    score: Optional[float] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceDocumentSchema] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
