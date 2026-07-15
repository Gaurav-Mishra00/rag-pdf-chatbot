import sqlite3
import os
import logging
from contextlib import contextmanager
from app.core.config import settings

logger = logging.getLogger(__name__)


@contextmanager
def get_db_connection():
    """
    Context manager that yields a database connection.
    Enables dictionary-like row access, enforces foreign key constraints,
    and handles automatic transaction commit/rollback.

    Concurrency notes:
    - WAL mode is enabled so readers never block writers and vice-versa.
    - busy_timeout=5000ms prevents immediate SQLITE_BUSY errors under
      concurrent request load; after 5 s the error propagates normally.
    - This is sufficient for single-process multi-threaded deployments
      (e.g. Uvicorn with a single worker).  Multi-process deployments
      (--workers > 1) require an external database such as PostgreSQL.
    """
    db_path = settings.SQLITE_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    # check_same_thread=False is safe here because each call creates its own connection
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Enable WAL mode and set a busy timeout for concurrent-write resilience
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys = ON;")

    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Database transaction rolled back due to error: %s", exc)
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Creates necessary database tables if they do not exist.
    """
    logger.info("Initializing SQLite database at: %s", settings.SQLITE_DB_PATH)
    with get_db_connection() as conn:
        # 1. Documents Metadata Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT UNIQUE,
                file_path TEXT,
                file_size_bytes INTEGER,
                chunk_count INTEGER,
                status TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. FAISS Vector Store Chunk Map Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
            );
        """)

        # 3. Conversational Chat History Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    logger.info("Database tables initialized successfully.")
