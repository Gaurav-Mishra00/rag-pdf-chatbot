import hashlib
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.core.config import settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_user_id_from_api_key(api_key: str) -> str:
    """
    Derives a unique user_id by hashing the API key (SHA-256).
    Ensures raw credentials are not stored in database fields.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
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

    # Allow comma-separated multiple keys in config
    valid_keys = [k.strip() for k in settings.API_KEY.split(",") if k.strip()]

    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )

    # Return the SHA-256 hashed user ID of the key
    return get_user_id_from_api_key(api_key)
