import anyio
from typing import Dict, List
from app.core.database import get_db_connection


def _db_list_sessions(user_id: str, limit: int = 10, offset: int = 0) -> List[Dict]:
    """Sync DB call: returns all distinct sessions for user_id with message count and last activity (paginated)."""
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                session_id,
                COUNT(*) AS message_count,
                MAX(created_at) AS last_activity
            FROM chat_history
            WHERE user_id = ?
            GROUP BY session_id
            ORDER BY last_activity DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, limit, offset)
        ).fetchall()
        return [dict(row) for row in rows]


def _db_get_total_sessions(user_id: str) -> int:
    """Sync DB call: returns total number of unique sessions owned by user_id."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(DISTINCT session_id) AS cnt FROM chat_history WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return row["cnt"] if row else 0


def _db_get_message_count(session_id: str, user_id: str) -> int:
    """Sync DB call: returns total message count for a session owned by user_id."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM chat_history WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        return row["cnt"] if row else 0


def _db_get_history(session_id: str, user_id: str) -> List[Dict[str, str]]:
    """Sync database call to retrieve chat history for session owned by user_id."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? AND user_id = ? ORDER BY id ASC",
            (session_id, user_id)
        )
        return [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]


def _db_add_message(session_id: str, role: str, content: str, user_id: str) -> None:
    """Sync database call to add a message to history under user_id."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO chat_history (session_id, role, content, user_id) VALUES (?, ?, ?, ?)",
            (session_id, role, content, user_id)
        )


def _db_clear_history(session_id: str, user_id: str) -> None:
    """Sync database call to clear chat history for session owned by user_id."""
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM chat_history WHERE session_id = ? AND user_id = ?",
            (session_id, user_id)
        )


class HistoryManager:
    """
    Manages conversational history for RAG chatbot sessions.
    Backed by a persistent SQLite database, with non-blocking async access 
    achieved using a thread pool via `anyio.to_thread.run_sync`.
    """

    async def get_history(self, session_id: str, user_id: str = "default_user") -> List[Dict[str, str]]:
        """
        Retrieves the turn-history list of message dictionaries for a session ID and user_id.
        """
        return await anyio.to_thread.run_sync(_db_get_history, session_id, user_id)

    async def add_message(self, session_id: str, role: str, content: str, user_id: str = "default_user") -> None:
        """
        Appends a message role and text content to the session's history under user_id.
        """
        await anyio.to_thread.run_sync(_db_add_message, session_id, role, content, user_id)

    async def clear_history(self, session_id: str, user_id: str = "default_user") -> None:
        """
        Clears the stored conversational history for a session ID and user_id.
        """
        await anyio.to_thread.run_sync(_db_clear_history, session_id, user_id)

    async def list_sessions(self, user_id: str = "default_user", limit: int = 10, offset: int = 0) -> List[Dict]:
        """
        Returns all distinct session IDs for user_id with message count and last activity timestamp.
        Each dict contains: session_id, message_count, last_activity. Supports pagination.
        """
        return await anyio.to_thread.run_sync(_db_list_sessions, user_id, limit, offset)

    async def get_total_sessions_count(self, user_id: str = "default_user") -> int:
        """
        Returns the total number of unique sessions for the given user_id.
        """
        return await anyio.to_thread.run_sync(_db_get_total_sessions, user_id)

    async def get_message_count(self, session_id: str, user_id: str = "default_user") -> int:
        """
        Returns the total number of messages stored for the given session and user_id.
        """
        return await anyio.to_thread.run_sync(_db_get_message_count, session_id, user_id)
