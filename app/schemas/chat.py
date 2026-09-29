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


from pydantic import BaseModel, Field, model_validator


class ChatQuery(BaseModel):
    message: str = Field(..., max_length=4000, description="The user prompt to process")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID")

    @model_validator(mode="before")
    @classmethod
    def resolve_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "message" not in data and "query" in data:
                data["message"] = data["query"]
        return data


class SourceDocumentSchema(BaseModel):
    document_name: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: str
    similarity_score: Optional[float] = None
    rerank_score: Optional[float] = None
    score: Optional[float] = None
    chunk_id: Optional[str] = None


class ChatResponseMetadata(BaseModel):
    """Typed metadata returned with every chat response."""
    model_name: str
    llm_provider: str
    embeddings_provider: str


class PipelineTimings(BaseModel):
    """Timing breakdown for each pipeline stage in milliseconds."""
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceDocumentSchema] = Field(default_factory=list)
    metadata: ChatResponseMetadata
    relevance_passed: bool = True
    timings: Optional[PipelineTimings] = None


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


# ---------------------------------------------------------------------------
# Evaluation schemas
# ---------------------------------------------------------------------------

class EvaluationQuery(BaseModel):
    """Request body for the /evaluate endpoint."""
    question: str = Field(..., max_length=4000, description="The question to evaluate retrieval for")
    expected_answer: Optional[str] = Field(None, description="Optional ground-truth answer for comparison")
    expected_sources: Optional[List[str]] = Field(None, description="Optional list of expected source filenames")


class RetrievalMetrics(BaseModel):
    """Metrics about the retrieval stage."""
    candidates_retrieved: int = 0
    candidates_after_rerank: int = 0
    mean_similarity_score: float = 0.0
    max_similarity_score: float = 0.0
    min_similarity_score: float = 0.0
    mean_rerank_score: Optional[float] = None
    relevance_passed: bool = True
    source_coverage: Optional[float] = None


class AnswerMetrics(BaseModel):
    """Metrics about the answer quality."""
    answer_length: int = 0
    citation_count: int = 0
    unique_sources_cited: int = 0
    has_i_dont_know: bool = False


class EvaluationResponse(BaseModel):
    """Response for the /evaluate endpoint."""
    question: str
    answer: str
    retrieval_metrics: RetrievalMetrics
    answer_metrics: AnswerMetrics
    timings: PipelineTimings
    sources: List[SourceDocumentSchema] = Field(default_factory=list)
