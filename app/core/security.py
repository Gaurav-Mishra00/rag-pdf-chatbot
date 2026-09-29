import contextvars
import hashlib
import secrets
from fastapi import Security, HTTPException, status, Request
from fastapi.security.api_key import APIKeyHeader
from app.core.config import settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Request-scoped context variable for the authenticated user ID
current_user_id = contextvars.ContextVar("current_user_id", default=None)


def get_user_id_from_api_key(api_key: str) -> str:
    """
    Derives a unique user_id by hashing the API key (SHA-256).
    Ensures raw credentials are not stored in database fields.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


async def verify_api_key(
    request: Request,
    api_key: str = Security(api_key_header)
) -> str:
    """
    Dependency to verify API keys for endpoint protection.
    Allows single or comma-separated API keys in settings.
    Returns the SHA-256 hashed user_id to ensure tenant isolation.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
        )

    # Allow comma-separated multiple keys in config, plus configured provider keys for convenience
    valid_keys = [k.strip() for k in settings.API_KEY.split(",") if k.strip()]
    if settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY.strip():
        valid_keys.append(settings.GOOGLE_API_KEY.strip())
    if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip():
        valid_keys.append(settings.OPENAI_API_KEY.strip())
    if settings.ANTHROPIC_API_KEY and settings.ANTHROPIC_API_KEY.strip():
        valid_keys.append(settings.ANTHROPIC_API_KEY.strip())

    if not any(secrets.compare_digest(api_key, vk) for vk in valid_keys):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials. Please verify your API key in Settings matches API_KEY in .env.",
        )

    # Return the SHA-256 hashed user ID of the key
    user_id = get_user_id_from_api_key(api_key)
    
    # Store user_id in the request state for rate limiting
    request.state.user_id = user_id
    
    # Store user_id in the contextvar for request-scoped access
    current_user_id.set(user_id)
    
    return user_id
