# tests/test_rate_limiter.py
"""Tests for rate limiting functionality."""

import time
import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limiter_allows_requests_under_limit(self):
        """Requests under limit should be allowed."""
        from rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=10, enabled=True)
        limiter.reset()

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get.return_value = None

        # First 10 requests should be allowed
        for _ in range(10):
            is_limited, remaining = limiter.is_rate_limited(mock_request)
            assert not is_limited
            limiter.record_request(mock_request)

    def test_rate_limiter_blocks_when_limit_exceeded(self):
        """Requests over limit should be blocked."""
        from rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=5, enabled=True)
        limiter.reset()

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get.return_value = None

        # Make 5 requests
        for _ in range(5):
            limiter.record_request(mock_request)

        # 6th request should be limited
        is_limited, remaining = limiter.is_rate_limited(mock_request)
        assert is_limited
        assert remaining == 0

    def test_rate_limiter_tracks_remaining(self):
        """Should correctly track remaining requests."""
        from rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=10, enabled=True)
        limiter.reset()

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get.return_value = None

        # Check initial remaining
        _, remaining = limiter.is_rate_limited(mock_request)
        assert remaining == 10

        # Make 3 requests
        for _ in range(3):
            limiter.record_request(mock_request)

        # Should have 7 remaining
        _, remaining = limiter.is_rate_limited(mock_request)
        assert remaining == 7

    def test_rate_limiter_disabled(self):
        """Disabled rate limiter should allow all requests."""
        from rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=1, enabled=False)
        limiter.reset()

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get.return_value = None

        # Make many requests
        for _ in range(100):
            is_limited, remaining = limiter.is_rate_limited(mock_request)
            assert not is_limited
            limiter.record_request(mock_request)

    def test_rate_limiter_per_ip(self):
        """Rate limiting should be per IP address."""
        from rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, enabled=True)
        limiter.reset()

        mock_request_1 = MagicMock()
        mock_request_1.client.host = "192.168.1.1"
        mock_request_1.headers.get.return_value = None

        mock_request_2 = MagicMock()
        mock_request_2.client.host = "192.168.1.2"
        mock_request_2.headers.get.return_value = None

        # Exhaust limit for IP 1
        limiter.record_request(mock_request_1)
        limiter.record_request(mock_request_1)
        is_limited, _ = limiter.is_rate_limited(mock_request_1)
        assert is_limited

        # IP 2 should still have full quota
        is_limited, remaining = limiter.is_rate_limited(mock_request_2)
        assert not is_limited
        assert remaining == 2

    def test_rate_limiter_uses_x_forwarded_for(self):
        """Should use X-Forwarded-For header when present."""
        from rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, enabled=True)
        limiter.reset()

        mock_request = MagicMock()
        mock_request.client.host = "10.0.0.1"
        mock_request.headers.get.return_value = "203.0.113.50, 70.41.3.18"

        limiter.record_request(mock_request)
        limiter.record_request(mock_request)

        # Should be limited based on the X-Forwarded-For IP
        is_limited, _ = limiter.is_rate_limited(mock_request)
        assert is_limited

    def test_check_rate_limit_raises_exception(self):
        """check_rate_limit should raise HTTPException when limited."""
        from rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=1, enabled=True)
        limiter.reset()

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get.return_value = None

        # First request should pass
        limiter.check_rate_limit(mock_request)

        # Second request should raise
        with pytest.raises(HTTPException) as exc_info:
            limiter.check_rate_limit(mock_request)

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail

    def test_rate_limiter_reset(self):
        """reset() should clear all counters."""
        from rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=1, enabled=True)

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get.return_value = None

        # Exhaust limit
        limiter.record_request(mock_request)
        is_limited, _ = limiter.is_rate_limited(mock_request)
        assert is_limited

        # Reset
        limiter.reset()

        # Should no longer be limited
        is_limited, _ = limiter.is_rate_limited(mock_request)
        assert not is_limited

    def test_global_rate_limiters_exist(self):
        """Global rate limiter instances should be defined."""
        from rate_limiter import api_rate_limiter, upload_rate_limiter

        assert api_rate_limiter is not None
        assert upload_rate_limiter is not None
        assert api_rate_limiter.requests_per_minute == 100
        assert upload_rate_limiter.requests_per_minute == 10
