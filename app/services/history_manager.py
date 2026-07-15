import anyio
from typing import Dict, List
from app.core.database import get_db_connection


def _db_list_sessions() -> List[Dict]:
    """Sync DB call: returns all distinct sessions with message count and last activity."""
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                session_id,
                COUNT(*) AS message_count,
                MAX(created_at) AS last_activity
            FROM chat_history
            GROUP BY session_id
            ORDER BY last_activity DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def _db_get_message_count(session_id: str) -> int:
    """Sync DB call: returns total message count for a session."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM chat_history WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["cnt"] if row else 0


def _db_get_history(session_id: str) -> List[Dict[str, str]]:
    """Sync database call to retrieve chat history."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        return [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]


def _db_add_message(session_id: str, role: str, content: str) -> None:
    """Sync database call to add a message to history."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )


def _db_clear_history(session_id: str) -> None:
    """Sync database call to clear chat history."""
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM chat_history WHERE session_id = ?",
            (session_id,)
        )


class HistoryManager:
    """
    Manages conversational history for RAG chatbot sessions.
    Backed by a persistent SQLite database, with non-blocking async access 
    achieved using a thread pool via `anyio.to_thread.run_sync`.
    """

    async def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Retrieves the turn-history list of message dictionaries for a session ID.
        """
        return await anyio.to_thread.run_sync(_db_get_history, session_id)

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Appends a message role and text content to the session's history.
        """
        await anyio.to_thread.run_sync(_db_add_message, session_id, role, content)

    async def clear_history(self, session_id: str) -> None:
        """
        Clears the stored conversational history for a session ID.
        """
        await anyio.to_thread.run_sync(_db_clear_history, session_id)

    async def list_sessions(self) -> List[Dict]:
        """
        Returns all distinct session IDs with message count and last activity timestamp.
        Each dict contains: session_id, message_count, last_activity.
        """
        return await anyio.to_thread.run_sync(_db_list_sessions)

    async def get_message_count(self, session_id: str) -> int:
        """
        Returns the total number of messages stored for the given session.
        """
        return await anyio.to_thread.run_sync(_db_get_message_count, session_id)
