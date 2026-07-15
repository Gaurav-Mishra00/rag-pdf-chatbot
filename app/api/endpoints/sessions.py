import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_history_manager
from app.core.security import verify_api_key
from app.schemas.chat import (
    MessageRole,
    Message,
    SessionListResponse,
    SessionSummary,
    SessionHistoryResponse,
)
from app.services.history_manager import HistoryManager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "",
    response_model=SessionListResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def list_sessions(
    history_manager: HistoryManager = Depends(get_history_manager),
) -> SessionListResponse:
    """
    Lists all active chat sessions with their message count and last
    activity timestamp, ordered by most recent first.
    """
    raw = await history_manager.list_sessions()
    sessions = [
        SessionSummary(
            session_id=row["session_id"],
            message_count=row["message_count"],
            last_activity=row.get("last_activity"),
        )
        for row in raw
    ]
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get(
    "/{session_id}",
    response_model=SessionHistoryResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def get_session_history(
    session_id: str,
    history_manager: HistoryManager = Depends(get_history_manager),
) -> SessionHistoryResponse:
    """
    Returns the complete message history for a specific session.
    Raises 404 if the session has no recorded history.
    """
    raw_history = await history_manager.get_history(session_id)
    if not raw_history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found or has no history.",
        )

    messages = [
        Message(role=MessageRole(turn["role"]), content=turn["content"])
        for turn in raw_history
    ]
    return SessionHistoryResponse(
        session_id=session_id,
        messages=messages,
        total=len(messages),
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def delete_session(
    session_id: str,
    history_manager: HistoryManager = Depends(get_history_manager),
) -> dict:
    """
    Clears all stored message history for a session.
    Idempotent — deleting a non-existent session returns 200 with a note.
    """
    count_before = await history_manager.get_message_count(session_id)
    await history_manager.clear_history(session_id)

    if count_before == 0:
        return {
            "message": "Session had no history; nothing to delete.",
            "session_id": session_id,
            "messages_deleted": 0,
        }

    logger.info("Cleared %d messages for session '%s'.", count_before, session_id)
    return {
        "message": "Session history cleared successfully.",
        "session_id": session_id,
        "messages_deleted": count_before,
    }
