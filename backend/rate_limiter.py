"""Simple in-memory rate limiter for API endpoints."""

import time
from collections import defaultdict
from typing import Dict, Tuple

from fastapi import HTTPException, Request


class RateLimiter:
    """In-memory rate limiter using sliding window algorithm."""

    def __init__(self, requests_per_minute: int = 100, enabled: bool = True):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute per IP.
            enabled: Whether rate limiting is enabled.
        """
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self.window_size = 60
        self._requests: Dict[str, list] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request with X-Forwarded-For support.

        Args:
            request: FastAPI request object.

        Returns:
            Client IP address.
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_old_requests(self, ip: str, current_time: float) -> None:
        """Remove requests outside current window.

        Args:
            ip: Client IP address.
            current_time: Current timestamp.
        """
        cutoff = current_time - self.window_size
        self._requests[ip] = [
            ts for ts in self._requests[ip] if ts > cutoff
        ]

    def is_rate_limited(self, request: Request) -> Tuple[bool, int]:
        """Check if request should be rate limited.

        Args:
            request: Incoming FastAPI request.

        Returns:
            Tuple of is_limited boolean and requests_remaining count.
        """
        if not self.enabled:
            return False, self.requests_per_minute

        ip = self._get_client_ip(request)
        current_time = time.time()

        self._cleanup_old_requests(ip, current_time)

        request_count = len(self._requests[ip])

        if request_count >= self.requests_per_minute:
            return True, 0

        return False, self.requests_per_minute - request_count

    def record_request(self, request: Request) -> None:
        """Record request for rate limiting.

        Args:
            request: FastAPI request to record.
        """
        if not self.enabled:
            return

        ip = self._get_client_ip(request)
        self._requests[ip].append(time.time())

    def check_rate_limit(self, request: Request) -> None:
        """Check rate limit and raise HTTPException if exceeded.

        Args:
            request: The incoming FastAPI request.

        Raises:
            HTTPException: 429 Too Many Requests if rate limit exceeded.
        """
        is_limited, remaining = self.is_rate_limited(request)
        if is_limited:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": str(self.window_size)}
            )
        self.record_request(request)

    def reset(self) -> None:
        """Reset all rate limit counters."""
        self._requests.clear()


api_rate_limiter = RateLimiter(requests_per_minute=100, enabled=True)
upload_rate_limiter = RateLimiter(requests_per_minute=10, enabled=True)
