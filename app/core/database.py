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

        # Create performance optimization indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_session_user ON chat_history (session_id, user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id ON document_chunks (document_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_user ON documents (user_id);")

    logger.info("Database tables initialized successfully.")


def reconcile_storage_layers(vector_store, upload_dir: str) -> None:
    """
    Reconciles inconsistencies between SQLite metadata, the filesystem, and the FAISS index.
    Run on application startup to clean up orphaned assets from crash-between-steps failures.
    """
    logger.info("Running storage layers reconciliation...")

    # Get active chunks from FAISS (thread-safe)
    faiss_chunk_ids = set()
    if vector_store.vector_store is not None:
        faiss_chunk_ids = set(vector_store.vector_store.index_to_docstore_id.values())

    # Get state from SQLite
    try:
        with get_db_connection() as conn:
            # 1. Get all tracked chunk IDs and their parent document IDs
            chunk_rows = conn.execute("SELECT chunk_id, document_id FROM document_chunks").fetchall()
            db_chunk_to_doc = {row["chunk_id"]: row["document_id"] for row in chunk_rows}

            # 2. Get all tracked documents and their file paths
            doc_rows = conn.execute("SELECT document_id, file_path FROM documents").fetchall()
            db_doc_to_path = {row["document_id"]: row["file_path"] for row in doc_rows}
    except Exception as exc:
        logger.error("Reconciliation failed to read database: %s", exc)
        return

    # Check 1: SQLite chunk records missing from FAISS (e.g. crashed during delete)
    orphaned_db_docs = set()
    for chunk_id, doc_id in db_chunk_to_doc.items():
        if chunk_id not in faiss_chunk_ids:
            orphaned_db_docs.add(doc_id)

    if orphaned_db_docs:
        logger.warning(
            "Found %d documents in DB with missing FAISS vectors (orphan chunks). Deleting records...",
            len(orphaned_db_docs),
        )
        try:
            with get_db_connection() as conn:
                for doc_id in orphaned_db_docs:
                    # Deleting documents automatically cascades to document_chunks
                    conn.execute("DELETE FROM documents WHERE document_id = ?", (doc_id,))
                    # Also try to clean up the file if it exists
                    file_path = db_doc_to_path.get(doc_id)
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            logger.info("Cleaned up orphaned file: %s", file_path)
                        except Exception as e:
                            logger.error("Failed to delete orphaned file %s: %s", file_path, e)
        except Exception as exc:
            logger.error("Failed to clean up orphaned database records: %s", exc)

    # Check 2: FAISS chunks missing from SQLite (e.g. crashed during upload after FAISS write but before DB commit)
    db_chunk_ids = set(db_chunk_to_doc.keys())
    orphaned_faiss_chunks = [
        cid for cid in faiss_chunk_ids 
        if cid not in db_chunk_ids and cid != "initialization placeholder" and not cid.startswith("__")
    ]
    if orphaned_faiss_chunks:
        # Check actual metadata source to make sure we don't delete initial placeholder
        actual_orphans_to_delete = []
        for cid in orphaned_faiss_chunks:
            try:
                doc = vector_store.vector_store.docstore.search(cid)
                if doc and doc.metadata.get("source") == "__init__":
                    continue
                actual_orphans_to_delete.append(cid)
            except Exception:
                actual_orphans_to_delete.append(cid)

        if actual_orphans_to_delete:
            logger.warning(
                "Found %d orphaned vectors in FAISS missing from database. Removing from index...",
                len(actual_orphans_to_delete),
            )
            try:
                vector_store.delete_documents(actual_orphans_to_delete)
            except Exception as exc:
                logger.error("Failed to delete orphaned FAISS vectors: %s", exc)

    # Check 3: Files on disk missing from SQLite (e.g. crashed during upload after file save but before DB record)
    try:
        disk_files = []
        if os.path.exists(upload_dir):
            disk_files = [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if f.endswith(".pdf")]
        
        valid_paths = set(db_doc_to_path.values())
        for file_path in disk_files:
            norm_path = os.path.normpath(file_path)
            norm_valid_paths = {os.path.normpath(p) for p in valid_paths if p}
            if norm_path not in norm_valid_paths:
                logger.warning("Found orphaned file on disk: %s. Deleting...", file_path)
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error("Failed to delete orphaned file %s: %s", file_path, e)
    except Exception as exc:
        logger.error("Filesystem reconciliation failed: %s", exc)

    logger.info("Storage layers reconciliation complete.")
