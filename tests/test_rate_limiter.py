# tests/test_rate_limiter.py
"""Tests for rate limiting functionality."""

import time
from unittest.mock import MagicMock

import pytest
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


class TestRateLimiterMemoryCleanup:
    """Tests for rate limiter memory cleanup to prevent leaks."""

    def test_cleanup_stale_ips_removes_old_entries(self):
        """cleanup_stale_ips should remove IPs with no recent requests."""
        from rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=10, enabled=True)
        limiter.reset()

        # Simulate requests from multiple IPs
        mock_requests = []
        for i in range(5):
            mock_req = MagicMock()
            mock_req.client.host = f"192.168.1.{i}"
            mock_req.headers.get.return_value = None
            mock_requests.append(mock_req)
            limiter.record_request(mock_req)

        # All 5 IPs should be tracked
        assert len(limiter._requests) == 5

        # Manually age all entries to be older than window
        old_time = time.time() - 120  # 2 minutes ago
        for ip in limiter._requests:
            limiter._requests[ip] = [old_time]

        # Run stale cleanup
        limiter.cleanup_stale_ips()

        # All entries should be removed since they're stale
        assert len(limiter._requests) == 0

    def test_cleanup_stale_ips_preserves_recent_entries(self):
        """cleanup_stale_ips should keep IPs with recent requests."""
        from rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=10, enabled=True)
        limiter.reset()

        # Add recent request
        mock_req1 = MagicMock()
        mock_req1.client.host = "192.168.1.1"
        mock_req1.headers.get.return_value = None
        limiter.record_request(mock_req1)

        # Add old request from different IP
        mock_req2 = MagicMock()
        mock_req2.client.host = "192.168.1.2"
        mock_req2.headers.get.return_value = None
        limiter.record_request(mock_req2)

        # Age the second IP's entry
        limiter._requests["192.168.1.2"] = [time.time() - 120]

        # Cleanup
        limiter.cleanup_stale_ips()

        # Only the recent IP should remain
        assert "192.168.1.1" in limiter._requests
        assert "192.168.1.2" not in limiter._requests
        assert len(limiter._requests) == 1

    def test_cleanup_stale_ips_removes_empty_entries(self):
        """cleanup_stale_ips should remove IPs with empty request lists."""
        from rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=10, enabled=True)
        limiter.reset()

        # Manually add empty entries (could happen after cleanup_old_requests)
        limiter._requests["192.168.1.1"] = []
        limiter._requests["192.168.1.2"] = []
        limiter._requests["192.168.1.3"] = [time.time()]  # Recent

        assert len(limiter._requests) == 3

        limiter.cleanup_stale_ips()

        # Only the non-empty entry should remain
        assert len(limiter._requests) == 1
        assert "192.168.1.3" in limiter._requests

    def test_periodic_cleanup_called_on_check(self):
        """Stale IP cleanup should be triggered periodically during checks."""
        from rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=10, enabled=True)
        limiter.reset()

        # Set last cleanup time to the past
        limiter._last_full_cleanup = time.time() - 120  # 2 minutes ago

        # Add a stale IP entry
        limiter._requests["192.168.1.100"] = [time.time() - 120]

        # Make a request from a different IP - this should trigger cleanup
        mock_req = MagicMock()
        mock_req.client.host = "192.168.1.1"
        mock_req.headers.get.return_value = None

        # Check should trigger periodic cleanup
        limiter.check_rate_limit(mock_req)

        # The stale IP should have been cleaned up
        assert "192.168.1.100" not in limiter._requests


class TestWebSocketRateLimiter:
    """Tests for WebSocket connection rate limiting."""

    def test_websocket_rate_limiter_exists(self):
        """WebSocket rate limiter should be available."""
        from rate_limiter import websocket_rate_limiter

        assert websocket_rate_limiter is not None

    def test_websocket_rate_limiter_tracks_connections(self):
        """WebSocket rate limiter should track connections per IP."""
        from rate_limiter import WebSocketRateLimiter

        limiter = WebSocketRateLimiter(max_connections_per_ip=3, enabled=True)
        limiter.reset()

        ip = "192.168.1.1"

        # First 3 connections should succeed
        assert limiter.can_connect(ip) is True
        limiter.record_connection(ip)
        assert limiter.can_connect(ip) is True
        limiter.record_connection(ip)
        assert limiter.can_connect(ip) is True
        limiter.record_connection(ip)

        # 4th connection should fail
        assert limiter.can_connect(ip) is False

        # Release one connection
        limiter.release_connection(ip)

        # Now should be able to connect again
        assert limiter.can_connect(ip) is True

    def test_websocket_rate_limiter_disabled(self):
        """When disabled, WebSocket rate limiter should allow all connections."""
        from rate_limiter import WebSocketRateLimiter

        limiter = WebSocketRateLimiter(max_connections_per_ip=1, enabled=False)
        ip = "192.168.1.1"

        # Even if we record many connections, should always allow
        for _ in range(10):
            assert limiter.can_connect(ip) is True
            limiter.record_connection(ip)

    def test_websocket_rate_limiter_tracks_per_ip(self):
        """WebSocket rate limiter should track each IP separately."""
        from rate_limiter import WebSocketRateLimiter

        limiter = WebSocketRateLimiter(max_connections_per_ip=2, enabled=True)
        limiter.reset()

        ip1 = "192.168.1.1"
        ip2 = "192.168.1.2"

        # Fill up IP1
        limiter.record_connection(ip1)
        limiter.record_connection(ip1)
        assert limiter.can_connect(ip1) is False

        # IP2 should still be able to connect
        assert limiter.can_connect(ip2) is True
        limiter.record_connection(ip2)
