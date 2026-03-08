"""Optional API key authentication for EchoScribe."""

import logging
import os
import secrets
import time
from typing import Any, Dict, Optional

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from backend.config_manager import get_config

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

_auth_config_override: Optional[Dict[str, Any]] = None
_ws_auth_tokens: Dict[str, float] = {}
WS_AUTH_TOKEN_TTL_SECONDS = 120


def get_auth_config() -> Dict[str, Any]:
    """Get authentication configuration.

    Returns:
        Authentication config dict with enabled and api_key keys.
    """
    if _auth_config_override is not None:
        return _auth_config_override

    config = get_config()
    return config.get("auth", {"enabled": False, "api_key": ""})


def _get_effective_api_key() -> str:
    """Get effective API key with environment variable precedence.

    Returns:
        API key to use for authentication.
    """
    env_key = os.environ.get("ECHOSCRIBE_API_KEY")
    if env_key:
        return env_key

    auth_config = get_auth_config()
    return auth_config.get("api_key", "")


def verify_api_key(api_key: Optional[str]) -> bool:
    """Verify if provided API key is valid.

    Args:
        api_key: API key to verify, or None if not provided.

    Returns:
        True if authentication passes, False otherwise.
    """
    auth_config = get_auth_config()

    if not auth_config.get("enabled", False):
        return True

    expected_key = _get_effective_api_key()

    if not expected_key:
        logger.warning("Auth is enabled but no API key is configured")
        return False

    if not api_key:
        return False

    return _secure_compare(api_key, expected_key)


def _secure_compare(a: str, b: str) -> bool:
    """Perform constant-time string comparison to prevent timing attacks.

    Args:
        a: First string to compare.
        b: Second string to compare.

    Returns:
        True if strings are equal, False otherwise.
    """
    import hmac

    return hmac.compare_digest(a.encode(), b.encode())


async def api_key_auth(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """FastAPI dependency for API key authentication.

    Args:
        api_key: API key from X-API-Key header.

    Returns:
        API key if valid, None if auth is disabled.

    Raises:
        HTTPException: 401 if authentication fails.
    """
    if verify_api_key(api_key):
        return api_key

    logger.warning("Authentication failed - invalid or missing API key")
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )


def _prune_expired_ws_tokens(now: Optional[float] = None) -> None:
    """Remove expired WebSocket auth tokens."""
    current_time = now if now is not None else time.time()
    expired_tokens = [
        token
        for token, expires_at in _ws_auth_tokens.items()
        if expires_at <= current_time
    ]
    for token in expired_tokens:
        _ws_auth_tokens.pop(token, None)


def issue_ws_auth_token(api_key: Optional[str]) -> str:
    """Issue short-lived one-time token for WebSocket authentication."""
    if not verify_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    token = secrets.token_urlsafe(32)
    now = time.time()
    _prune_expired_ws_tokens(now)
    _ws_auth_tokens[token] = now + WS_AUTH_TOKEN_TTL_SECONDS
    return token


def consume_ws_auth_token(token: Optional[str]) -> bool:
    """Validate and consume a one-time WebSocket authentication token."""
    if not get_auth_config().get("enabled", False):
        return True
    if not token:
        return False

    now = time.time()
    _prune_expired_ws_tokens(now)
    expires_at = _ws_auth_tokens.pop(token, None)
    return expires_at is not None and expires_at > now


def reset_ws_auth_tokens() -> None:
    """Clear in-memory WebSocket authentication tokens (test helper)."""
    _ws_auth_tokens.clear()
