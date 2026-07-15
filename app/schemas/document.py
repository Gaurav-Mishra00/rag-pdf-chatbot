from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    filename: str
    file_size_bytes: int
    content_type: str


class DocumentUploadResponse(BaseModel):
    filename: str
    status: IngestionStatus
    message: str
    document_id: str


class DocumentStatusResponse(BaseModel):
    """Full metadata for a single stored document."""
    document_id: str
    filename: str
    status: IngestionStatus
    chunk_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None


class SearchQuerySchema(BaseModel):
    query: str
    top_k: int = Field(default=4, ge=1, le=20)


class SearchResultDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]
    score: float
