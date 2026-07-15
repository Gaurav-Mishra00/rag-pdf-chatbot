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
    Performs migrations to add user_id column and isolate unique constraints.
    """
    logger.info("Initializing SQLite database at: %s", settings.SQLITE_DB_PATH)
    with get_db_connection() as conn:
        # Create documents table with multi-tenant unique constraint if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT,
                file_path TEXT,
                file_size_bytes INTEGER,
                chunk_count INTEGER,
                status TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT DEFAULT 'default_user',
                UNIQUE(filename, user_id)
            );
        """)

        # Migration: Check if we need to migrate the existing documents table to add user_id
        cursor = conn.execute("PRAGMA table_info(documents);")
        columns = [row["name"] for row in cursor.fetchall()]
        if "user_id" not in columns:
            logger.info("Migrating 'documents' table to add user_id and unique constraint...")
            conn.execute("ALTER TABLE documents RENAME TO documents_old;")
            conn.execute("""
                CREATE TABLE documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT,
                    file_path TEXT,
                    file_size_bytes INTEGER,
                    chunk_count INTEGER,
                    status TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT DEFAULT 'default_user',
                    UNIQUE(filename, user_id)
                );
            """)
            conn.execute("""
                INSERT INTO documents (document_id, filename, file_path, file_size_bytes, chunk_count, status, error_message, created_at, user_id)
                SELECT document_id, filename, file_path, file_size_bytes, chunk_count, status, error_message, created_at, 'default_user'
                FROM documents_old;
            """)
            conn.execute("DROP TABLE documents_old;")
            logger.info("'documents' table migration completed.")

        # 2. FAISS Vector Store Chunk Map Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
            );
        """)

        # 3. Conversational Chat History Table (with user_id)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT DEFAULT 'default_user'
            );
        """)

        # Migration: Check if we need to add user_id to existing chat_history table
        cursor = conn.execute("PRAGMA table_info(chat_history);")
        columns = [row["name"] for row in cursor.fetchall()]
        if "user_id" not in columns:
            logger.info("Migrating 'chat_history' table to add user_id...")
            conn.execute("ALTER TABLE chat_history ADD COLUMN user_id TEXT DEFAULT 'default_user';")
            logger.info("'chat_history' table migration completed.")

    logger.info("Database tables initialized successfully.")
