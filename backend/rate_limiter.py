"""Simple in-memory rate limiter for API endpoints."""

import time
from collections import defaultdict
from typing import Dict, Tuple

from fastapi import HTTPException, Request


class RateLimiter:
    """In-memory rate limiter using sliding window algorithm."""

    # How often to run full cleanup (in seconds)
    CLEANUP_INTERVAL = 60

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
        self._last_full_cleanup = time.time()

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
        self._requests[ip] = [ts for ts in self._requests[ip] if ts > cutoff]

    def cleanup_stale_ips(self) -> int:
        """Remove IPs that have no recent requests to prevent memory leaks.

        This should be called periodically to clean up IPs that have stopped
        making requests. IPs with no timestamps or only stale timestamps are removed.

        Returns:
            Number of IPs removed.
        """
        current_time = time.time()
        cutoff = current_time - self.window_size
        stale_ips = []

        for ip, timestamps in self._requests.items():
            # Remove stale timestamps
            recent = [ts for ts in timestamps if ts > cutoff]
            if not recent:
                # No recent requests, mark for removal
                stale_ips.append(ip)
            else:
                # Update with only recent timestamps
                self._requests[ip] = recent

        # Remove stale IPs
        for ip in stale_ips:
            del self._requests[ip]

        self._last_full_cleanup = current_time
        return len(stale_ips)

    def _maybe_cleanup_stale_ips(self) -> None:
        """Run stale IP cleanup if enough time has passed since last cleanup."""
        current_time = time.time()
        if current_time - self._last_full_cleanup >= self.CLEANUP_INTERVAL:
            self.cleanup_stale_ips()

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
        # Periodically clean up stale IPs to prevent memory leaks
        self._maybe_cleanup_stale_ips()

        is_limited, remaining = self.is_rate_limited(request)
        if is_limited:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": str(self.window_size)},
            )
        self.record_request(request)

    def reset(self) -> None:
        """Reset all rate limit counters."""
        self._requests.clear()
        self._last_full_cleanup = time.time()


api_rate_limiter = RateLimiter(requests_per_minute=100, enabled=True)
upload_rate_limiter = RateLimiter(requests_per_minute=10, enabled=True)


class WebSocketRateLimiter:
    """Rate limiter for WebSocket connections per IP address."""

    def __init__(self, max_connections_per_ip: int = 5, enabled: bool = True):
        """Initialize WebSocket rate limiter.

        Args:
            max_connections_per_ip: Maximum concurrent WebSocket connections per IP.
            enabled: Whether rate limiting is enabled.
        """
        self.max_connections_per_ip = max_connections_per_ip
        self.enabled = enabled
        self._connections: Dict[str, int] = defaultdict(int)

    def can_connect(self, ip: str) -> bool:
        """Check if IP can open a new WebSocket connection.

        Args:
            ip: Client IP address.

        Returns:
            True if connection is allowed, False if limit exceeded.
        """
        if not self.enabled:
            return True
        return self._connections[ip] < self.max_connections_per_ip

    def record_connection(self, ip: str) -> None:
        """Record a new WebSocket connection from IP.

        Args:
            ip: Client IP address.
        """
        if not self.enabled:
            return
        self._connections[ip] += 1

    def release_connection(self, ip: str) -> None:
        """Release a WebSocket connection from IP.

        Args:
            ip: Client IP address.
        """
        if ip in self._connections and self._connections[ip] > 0:
            self._connections[ip] -= 1
            if self._connections[ip] == 0:
                del self._connections[ip]

    def reset(self) -> None:
        """Reset all connection counters."""
        self._connections.clear()


websocket_rate_limiter = WebSocketRateLimiter(max_connections_per_ip=5, enabled=True)
