import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # FAISS Path
    FAISS_INDEX_PATH: str = "data/faiss_index"

    # Embedding Settings
    EMBEDDINGS_PROVIDER: Literal["openai", "huggingface", "google"] = "openai"
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"

    # LLM Settings
    LLM_PROVIDER: Literal["openai", "google", "anthropic"] = "openai"
    LLM_MODEL_NAME: str = "gpt-4o"
    TEMPERATURE: float = 0.0

    # API Keys
    OPENAI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global settings instance
settings = Settings()
