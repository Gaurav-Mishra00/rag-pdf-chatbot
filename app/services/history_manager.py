from typing import Dict, List


class HistoryManager:
    """
    Manages conversational history for RAG chatbot sessions.
    Currently utilizes an in-memory dictionary, making it easy to swap with
    SQLite, Redis, or PostgreSQL client stores.
    """

    def __init__(self):
        self._store: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Retrieves the turn-history list of message dictionaries for a session ID.
        """
        if session_id not in self._store:
            self._store[session_id] = []
        return self._store[session_id]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Appends a message role and text content to the session's history.
        """
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append({"role": role, "content": content})

    def clear_history(self, session_id: str) -> None:
        """
        Clears the stored conversational history for a session ID.
        """
        if session_id in self._store:
            self._store[session_id] = []
