import os
from typing import Literal
from pydantic import model_validator
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

    # FAISS Path
    FAISS_INDEX_PATH: str = "data/faiss_index"

    # SQLite DB and Upload Directory Settings
    SQLITE_DB_PATH: str = "data/db.sqlite3"
    UPLOAD_DIR: str = "data/uploads"

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

    @model_validator(mode="after")
    def resolve_file_secrets(self) -> "Settings":
        self.OPENAI_API_KEY = resolve_secret_value(self.OPENAI_API_KEY)
        self.GOOGLE_API_KEY = resolve_secret_value(self.GOOGLE_API_KEY)
        self.ANTHROPIC_API_KEY = resolve_secret_value(self.ANTHROPIC_API_KEY)
        return self


# Global settings instance
settings = Settings()
