import os
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def resolve_secret_value(value: str | None) -> str | None:
    """
    If value is prefixed with 'file://', reads the secret from the target path.
    Otherwise, returns the raw value directly. This supports secure secret mounting
    in cloud/production containerized orchestrations.
    """
    if value and value.startswith("file://"):
        secret_path = value[7:]
        try:
            if os.path.exists(secret_path):
                with open(secret_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            else:
                import logging
                logging.getLogger(__name__).warning("Secret file path '%s' not found.", secret_path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Failed to read secret file '%s': %s", secret_path, exc)
    return value


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "RAG Chatbot API"
    APP_ENV: str = "local"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    LOG_LEVEL: str = "INFO"

    # Security
    API_KEY: str = "change_me_in_production"
    ALLOWED_ORIGINS: str = "*"

    # FAISS Path
    FAISS_INDEX_PATH: str = "data/faiss_index"

    # SQLite DB and Upload Directory Settings
    SQLITE_DB_PATH: str = "data/db.sqlite3"
    UPLOAD_DIR: str = "data/uploads"

    # Embedding Settings
    EMBEDDINGS_PROVIDER: Literal["openai", "huggingface", "google"] = "huggingface"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = Field(default=2, ge=1, le=128)
    CHUNK_SIZE: int = Field(default=1200, ge=100, le=8000)
    CHUNK_OVERLAP: int = Field(default=180, ge=0)
    RETRIEVAL_K: int = Field(default=20, ge=1, le=100)
    CONTEXT_K: int = Field(default=4, ge=1, le=20)
    MIN_SIMILARITY: float = Field(default=0.35, ge=-1, le=1)
    RERANK_ENABLED: bool = True
    RERANK_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    RERANK_MIN_SCORE: float = Field(default=0.5, ge=0, le=1)
    ADAPTIVE_RERANK_THRESHOLD: float = Field(default=0.88, ge=0, le=1)

    # Efficiency & Performance Optimizations
    QUERY_CACHE_ENABLED: bool = True
    QUERY_CACHE_TTL_SECONDS: int = Field(default=3600, ge=1)
    QUERY_CACHE_MAX_SIZE: int = Field(default=500, ge=10)
    CHUNK_DEDUPLICATION_ENABLED: bool = True
    FAISS_INDEX_TYPE: Literal["flat", "hnsw"] = "flat"
    FAISS_HNSW_M: int = Field(default=32, ge=4, le=128)
    FAISS_HNSW_EF_SEARCH: int = Field(default=64, ge=8, le=512)
    FAISS_MMAP_ENABLED: bool = False

    # LLM Settings
    LLM_PROVIDER: Literal["openai", "google", "anthropic"] = "openai"
    LLM_MODEL_NAME: str = "gpt-4o"
    TEMPERATURE: float = 0.0
    LLM_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0)

    # API Keys
    OPENAI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @model_validator(mode="after")
    def resolve_file_secrets(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.CONTEXT_K > self.RETRIEVAL_K:
            raise ValueError("CONTEXT_K cannot exceed RETRIEVAL_K")
        self.OPENAI_API_KEY = resolve_secret_value(self.OPENAI_API_KEY)
        self.GOOGLE_API_KEY = resolve_secret_value(self.GOOGLE_API_KEY)
        self.ANTHROPIC_API_KEY = resolve_secret_value(self.ANTHROPIC_API_KEY)
        return self


# Global settings instance
settings = Settings()
