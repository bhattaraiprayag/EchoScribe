"""Optional API key authentication for EchoScribe."""

import logging
import os
from typing import Any, Dict, Optional

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from config_manager import get_config


logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

_auth_config_override: Optional[Dict[str, Any]] = None


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
    api_key: Optional[str] = Security(api_key_header)
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
        headers={"WWW-Authenticate": "ApiKey"}
    )
