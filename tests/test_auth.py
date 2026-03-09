# tests/test_auth.py

"""Tests for optional API key authentication middleware."""

from unittest.mock import patch

import pytest
from starlette.websockets import WebSocketDisconnect


class TestAuthConfig:
    """Test authentication configuration."""

    def test_auth_config_defaults(self):
        """Test that auth is disabled by default."""
        from backend.config_manager import get_config

        config = get_config()
        auth_config = config.get("auth", {})
        # Auth should be disabled by default
        assert auth_config.get("enabled", False) is False

    def test_auth_config_has_api_key_field(self):
        """Test that auth config has api_key field."""
        from backend.config_manager import get_config

        config = get_config()
        auth_config = config.get("auth", {})
        # api_key field should exist
        assert "api_key" in auth_config


class TestAuthMiddleware:
    """Test API key authentication middleware."""

    def test_get_auth_config_returns_defaults(self):
        """Test get_auth_config returns proper defaults."""
        from backend.auth import get_auth_config

        config = get_auth_config()
        assert "enabled" in config
        assert "api_key" in config
        assert isinstance(config["enabled"], bool)

    def test_verify_api_key_returns_true_when_auth_disabled(self):
        """Test that verify_api_key returns True when auth is disabled."""
        from backend.auth import verify_api_key

        # When auth is disabled (default), any key should be accepted
        assert verify_api_key(None) is True
        assert verify_api_key("") is True
        assert verify_api_key("any-key") is True

    def test_verify_api_key_returns_false_for_missing_key_when_enabled(
        self, monkeypatch
    ):
        """Test that verify_api_key returns False for missing key when
        auth is enabled."""
        from backend.auth import verify_api_key

        # Override auth config to enable auth
        monkeypatch.setattr(
            "backend.auth._auth_config_override",
            {"enabled": True, "api_key": "test-secret-key"},
        )

        assert verify_api_key(None) is False
        assert verify_api_key("") is False

        # Clean up
        monkeypatch.setattr("backend.auth._auth_config_override", None)

    def test_verify_api_key_returns_false_for_wrong_key_when_enabled(self, monkeypatch):
        """Test that verify_api_key returns False for wrong key when auth is enabled."""
        from backend.auth import verify_api_key

        # Override auth config to enable auth
        monkeypatch.setattr(
            "backend.auth._auth_config_override",
            {"enabled": True, "api_key": "test-secret-key"},
        )

        assert verify_api_key("wrong-key") is False

        # Clean up
        monkeypatch.setattr("backend.auth._auth_config_override", None)

    def test_verify_api_key_returns_true_for_correct_key_when_enabled(
        self, monkeypatch
    ):
        """Test that verify_api_key returns True for correct key when
        auth is enabled."""
        from backend.auth import verify_api_key

        # Override auth config to enable auth
        monkeypatch.setattr(
            "backend.auth._auth_config_override",
            {"enabled": True, "api_key": "test-secret-key"},
        )

        assert verify_api_key("test-secret-key") is True

        # Clean up
        monkeypatch.setattr("backend.auth._auth_config_override", None)


class TestAuthMiddlewareIntegration:
    """Integration tests for authentication middleware with FastAPI."""

    def test_api_allows_requests_when_auth_disabled(self, monkeypatch):
        """Test that API allows requests when auth is disabled."""
        from fastapi.testclient import TestClient

        from backend.auth import api_key_auth

        # Ensure auth is disabled
        monkeypatch.setattr(
            "backend.auth._auth_config_override", {"enabled": False, "api_key": ""}
        )

        # Create a simple test app
        from fastapi import Depends, FastAPI

        app = FastAPI()

        @app.get("/test")
        def test_endpoint(api_key: str = Depends(api_key_auth)):
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

        # Clean up
        monkeypatch.setattr("backend.auth._auth_config_override", None)

    def test_api_returns_401_when_auth_enabled_no_key(self, monkeypatch):
        """Test that API returns 401 when auth is enabled but no key provided."""
        from fastapi.testclient import TestClient

        from backend.auth import api_key_auth

        # Enable auth
        monkeypatch.setattr(
            "backend.auth._auth_config_override",
            {"enabled": True, "api_key": "secret-key"},
        )

        from fastapi import Depends, FastAPI

        app = FastAPI()

        @app.get("/test")
        def test_endpoint(api_key: str = Depends(api_key_auth)):
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 401
        assert "Invalid or missing API key" in response.json()["detail"]

        # Clean up
        monkeypatch.setattr("backend.auth._auth_config_override", None)

    def test_api_returns_401_when_auth_enabled_wrong_key(self, monkeypatch):
        """Test that API returns 401 when auth is enabled and wrong key provided."""
        from fastapi.testclient import TestClient

        from backend.auth import api_key_auth

        # Enable auth
        monkeypatch.setattr(
            "backend.auth._auth_config_override",
            {"enabled": True, "api_key": "secret-key"},
        )

        from fastapi import Depends, FastAPI

        app = FastAPI()

        @app.get("/test")
        def test_endpoint(api_key: str = Depends(api_key_auth)):
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test", headers={"X-API-Key": "wrong-key"})

        assert response.status_code == 401
        assert "Invalid or missing API key" in response.json()["detail"]

        # Clean up
        monkeypatch.setattr("backend.auth._auth_config_override", None)

    def test_api_allows_requests_when_auth_enabled_correct_key(self, monkeypatch):
        """Test that API allows requests when auth is enabled and correct
        key provided."""
        from fastapi.testclient import TestClient

        from backend.auth import api_key_auth

        # Enable auth
        monkeypatch.setattr(
            "backend.auth._auth_config_override",
            {"enabled": True, "api_key": "secret-key"},
        )

        from fastapi import Depends, FastAPI

        app = FastAPI()

        @app.get("/test")
        def test_endpoint(api_key: str = Depends(api_key_auth)):
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test", headers={"X-API-Key": "secret-key"})

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

        # Clean up
        monkeypatch.setattr("backend.auth._auth_config_override", None)

    def test_api_key_from_environment_variable(self, monkeypatch):
        """Test that API key can be loaded from environment variable."""

        # Set environment variable for API key
        monkeypatch.setenv("ECHOSCRIBE_API_KEY", "env-secret-key")

        # Override to enable auth but use env var for key
        monkeypatch.setattr(
            "backend.auth._auth_config_override",
            {
                "enabled": True,
                "api_key": "",  # Empty in config
            },
        )

        # The get_auth_config should check env var when api_key is empty
        from backend.auth import _get_effective_api_key

        effective_key = _get_effective_api_key()
        assert effective_key == "env-secret-key"

        # Clean up
        monkeypatch.setattr("backend.auth._auth_config_override", None)


class TestAuthEnvironmentVariable:
    """Test that API key can be set via environment variable."""

    def test_env_var_overrides_config(self, monkeypatch):
        """Test that ECHOSCRIBE_API_KEY env var overrides config value."""
        from backend.auth import _get_effective_api_key

        # Set env var
        monkeypatch.setenv("ECHOSCRIBE_API_KEY", "from-env")

        # Set config with different value
        monkeypatch.setattr(
            "backend.auth._auth_config_override",
            {"enabled": True, "api_key": "from-config"},
        )

        # Env var should take precedence
        assert _get_effective_api_key() == "from-env"

        # Clean up
        monkeypatch.setattr("backend.auth._auth_config_override", None)

    def test_config_used_when_no_env_var(self, monkeypatch):
        """Test that config api_key is used when env var is not set."""
        from backend.auth import _get_effective_api_key

        # Ensure env var is not set
        monkeypatch.delenv("ECHOSCRIBE_API_KEY", raising=False)

        # Set config value
        monkeypatch.setattr(
            "backend.auth._auth_config_override",
            {"enabled": True, "api_key": "from-config"},
        )

        # Config should be used
        assert _get_effective_api_key() == "from-config"

        # Clean up
        monkeypatch.setattr("backend.auth._auth_config_override", None)


class TestWebSocketAuthentication:
    """Test WebSocket authentication."""

    def test_websocket_rejects_connection_when_auth_enabled_no_token(
        self, sync_test_client, monkeypatch
    ):
        """WebSocket should reject config without auth token when auth is enabled."""
        monkeypatch.setattr(
            "backend.auth._auth_config_override",
            {"enabled": True, "api_key": "secret-key"},
        )
        try:
            with sync_test_client.websocket_connect("/ws/test-session") as websocket:
                websocket.send_json(
                    {"model": "tiny", "device": "cpu", "language": "en"}
                )
                payload = websocket.receive_json()
                assert payload["status"] == "error"
                assert payload["message"] == "Authentication failed"
                with pytest.raises(WebSocketDisconnect) as exc:
                    websocket.receive_json()
                assert exc.value.code == 4001
        finally:
            monkeypatch.setattr("backend.auth._auth_config_override", None)
            from backend.auth import reset_ws_auth_tokens

            reset_ws_auth_tokens()

    def test_websocket_rejects_connection_when_auth_enabled_bad_token(
        self, sync_test_client, monkeypatch
    ):
        """WebSocket should reject invalid auth tokens."""
        monkeypatch.setattr(
            "backend.auth._auth_config_override",
            {"enabled": True, "api_key": "secret-key"},
        )
        try:
            with sync_test_client.websocket_connect("/ws/test-session") as websocket:
                websocket.send_json(
                    {
                        "model": "tiny",
                        "device": "cpu",
                        "language": "en",
                        "auth_token": "bad-token",
                    }
                )
                payload = websocket.receive_json()
                assert payload["status"] == "error"
                assert payload["message"] == "Authentication failed"
                with pytest.raises(WebSocketDisconnect) as exc:
                    websocket.receive_json()
                assert exc.value.code == 4001
        finally:
            monkeypatch.setattr("backend.auth._auth_config_override", None)
            from backend.auth import reset_ws_auth_tokens

            reset_ws_auth_tokens()

    def test_websocket_accepts_connection_with_issued_token(
        self, sync_test_client, monkeypatch
    ):
        """WebSocket should accept a valid one-time token."""
        monkeypatch.setattr(
            "backend.auth._auth_config_override",
            {"enabled": True, "api_key": "secret-key"},
        )
        try:
            token_response = sync_test_client.post(
                "/api/ws-auth-token", headers={"X-API-Key": "secret-key"}
            )
            assert token_response.status_code == 200
            token = token_response.json()["token"]

            with patch("backend.main.is_model_loaded", return_value=True):
                with sync_test_client.websocket_connect(
                    "/ws/test-session"
                ) as websocket:
                    websocket.send_json(
                        {
                            "model": "tiny",
                            "device": "cpu",
                            "language": "en",
                            "auth_token": token,
                        }
                    )
                    first_status = websocket.receive_json()

            assert first_status["status"] == "ready"
        finally:
            monkeypatch.setattr("backend.auth._auth_config_override", None)
            from backend.auth import reset_ws_auth_tokens

            reset_ws_auth_tokens()

    def test_websocket_accepts_connection_when_auth_disabled(
        self, sync_test_client, monkeypatch
    ):
        """Test that WebSocket accepts connection when auth is disabled."""
        monkeypatch.setattr(
            "backend.auth._auth_config_override", {"enabled": False, "api_key": ""}
        )
        try:
            with patch("backend.main.is_model_loaded", return_value=True):
                with sync_test_client.websocket_connect(
                    "/ws/test-session"
                ) as websocket:
                    websocket.send_json(
                        {"model": "tiny", "device": "cpu", "language": "en"}
                    )
                    first_status = websocket.receive_json()
            assert first_status["status"] == "ready"
        finally:
            monkeypatch.setattr("backend.auth._auth_config_override", None)
