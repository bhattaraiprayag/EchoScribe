# tests/test_auth.py

"""Tests for optional API key authentication middleware."""

import os
import sys

import pytest

# Add backend directory to Python path
BACKEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"
)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


class TestAuthConfig:
    """Test authentication configuration."""

    def test_auth_config_defaults(self):
        """Test that auth is disabled by default."""
        from config_manager import get_config

        config = get_config()
        auth_config = config.get("auth", {})
        # Auth should be disabled by default
        assert auth_config.get("enabled", False) is False

    def test_auth_config_has_api_key_field(self):
        """Test that auth config has api_key field."""
        from config_manager import get_config

        config = get_config()
        auth_config = config.get("auth", {})
        # api_key field should exist
        assert "api_key" in auth_config


class TestAuthMiddleware:
    """Test API key authentication middleware."""

    def test_get_auth_config_returns_defaults(self):
        """Test get_auth_config returns proper defaults."""
        from auth import get_auth_config

        config = get_auth_config()
        assert "enabled" in config
        assert "api_key" in config
        assert isinstance(config["enabled"], bool)

    def test_verify_api_key_returns_true_when_auth_disabled(self):
        """Test that verify_api_key returns True when auth is disabled."""
        from auth import verify_api_key

        # When auth is disabled (default), any key should be accepted
        assert verify_api_key(None) is True
        assert verify_api_key("") is True
        assert verify_api_key("any-key") is True

    def test_verify_api_key_returns_false_for_missing_key_when_enabled(
        self, monkeypatch
    ):
        """Test that verify_api_key returns False for missing key when
        auth is enabled."""
        from auth import verify_api_key

        # Override auth config to enable auth
        monkeypatch.setattr(
            "auth._auth_config_override",
            {"enabled": True, "api_key": "test-secret-key"},
        )

        assert verify_api_key(None) is False
        assert verify_api_key("") is False

        # Clean up
        monkeypatch.setattr("auth._auth_config_override", None)

    def test_verify_api_key_returns_false_for_wrong_key_when_enabled(self, monkeypatch):
        """Test that verify_api_key returns False for wrong key when auth is enabled."""
        from auth import verify_api_key

        # Override auth config to enable auth
        monkeypatch.setattr(
            "auth._auth_config_override",
            {"enabled": True, "api_key": "test-secret-key"},
        )

        assert verify_api_key("wrong-key") is False

        # Clean up
        monkeypatch.setattr("auth._auth_config_override", None)

    def test_verify_api_key_returns_true_for_correct_key_when_enabled(
        self, monkeypatch
    ):
        """Test that verify_api_key returns True for correct key when
        auth is enabled."""
        from auth import verify_api_key

        # Override auth config to enable auth
        monkeypatch.setattr(
            "auth._auth_config_override",
            {"enabled": True, "api_key": "test-secret-key"},
        )

        assert verify_api_key("test-secret-key") is True

        # Clean up
        monkeypatch.setattr("auth._auth_config_override", None)


class TestAuthMiddlewareIntegration:
    """Integration tests for authentication middleware with FastAPI."""

    def test_api_allows_requests_when_auth_disabled(self, monkeypatch):
        """Test that API allows requests when auth is disabled."""
        from auth import api_key_auth
        from fastapi.testclient import TestClient

        # Ensure auth is disabled
        monkeypatch.setattr(
            "auth._auth_config_override", {"enabled": False, "api_key": ""}
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
        monkeypatch.setattr("auth._auth_config_override", None)

    def test_api_returns_401_when_auth_enabled_no_key(self, monkeypatch):
        """Test that API returns 401 when auth is enabled but no key provided."""
        from auth import api_key_auth
        from fastapi.testclient import TestClient

        # Enable auth
        monkeypatch.setattr(
            "auth._auth_config_override", {"enabled": True, "api_key": "secret-key"}
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
        monkeypatch.setattr("auth._auth_config_override", None)

    def test_api_returns_401_when_auth_enabled_wrong_key(self, monkeypatch):
        """Test that API returns 401 when auth is enabled and wrong key provided."""
        from auth import api_key_auth
        from fastapi.testclient import TestClient

        # Enable auth
        monkeypatch.setattr(
            "auth._auth_config_override", {"enabled": True, "api_key": "secret-key"}
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
        monkeypatch.setattr("auth._auth_config_override", None)

    def test_api_allows_requests_when_auth_enabled_correct_key(self, monkeypatch):
        """Test that API allows requests when auth is enabled and correct
        key provided."""
        from auth import api_key_auth
        from fastapi.testclient import TestClient

        # Enable auth
        monkeypatch.setattr(
            "auth._auth_config_override", {"enabled": True, "api_key": "secret-key"}
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
        monkeypatch.setattr("auth._auth_config_override", None)

    def test_api_key_from_environment_variable(self, monkeypatch):
        """Test that API key can be loaded from environment variable."""

        # Set environment variable for API key
        monkeypatch.setenv("ECHOSCRIBE_API_KEY", "env-secret-key")

        # Override to enable auth but use env var for key
        monkeypatch.setattr(
            "auth._auth_config_override",
            {
                "enabled": True,
                "api_key": "",  # Empty in config
            },
        )

        # The get_auth_config should check env var when api_key is empty
        from auth import _get_effective_api_key

        effective_key = _get_effective_api_key()
        assert effective_key == "env-secret-key"

        # Clean up
        monkeypatch.setattr("auth._auth_config_override", None)


class TestAuthEnvironmentVariable:
    """Test that API key can be set via environment variable."""

    def test_env_var_overrides_config(self, monkeypatch):
        """Test that ECHOSCRIBE_API_KEY env var overrides config value."""
        from auth import _get_effective_api_key

        # Set env var
        monkeypatch.setenv("ECHOSCRIBE_API_KEY", "from-env")

        # Set config with different value
        monkeypatch.setattr(
            "auth._auth_config_override", {"enabled": True, "api_key": "from-config"}
        )

        # Env var should take precedence
        assert _get_effective_api_key() == "from-env"

        # Clean up
        monkeypatch.setattr("auth._auth_config_override", None)

    def test_config_used_when_no_env_var(self, monkeypatch):
        """Test that config api_key is used when env var is not set."""
        from auth import _get_effective_api_key

        # Ensure env var is not set
        monkeypatch.delenv("ECHOSCRIBE_API_KEY", raising=False)

        # Set config value
        monkeypatch.setattr(
            "auth._auth_config_override", {"enabled": True, "api_key": "from-config"}
        )

        # Config should be used
        assert _get_effective_api_key() == "from-config"

        # Clean up
        monkeypatch.setattr("auth._auth_config_override", None)


class TestWebSocketAuthentication:
    """Test WebSocket authentication."""

    def test_websocket_rejects_connection_when_auth_enabled_no_key(self, monkeypatch):
        """Test that WebSocket rejects connection when auth is enabled but
        no key provided."""
        from unittest.mock import MagicMock, patch

        from fastapi.testclient import TestClient

        # Enable auth
        monkeypatch.setattr(
            "auth._auth_config_override", {"enabled": True, "api_key": "secret-key"}
        )

        # Mock torch.hub.load to avoid loading actual model
        with patch("torch.hub.load") as mock_hub_load:
            mock_hub_load.return_value = MagicMock()
            from main import app

            client = TestClient(app)

            # WebSocket without API key should be rejected
            with pytest.raises(Exception):  # noqa: B017
                with client.websocket_connect("/ws/test-session"):
                    pass

            # Should get a close or rejection
            # The exception type may vary, but connection should fail

        # Clean up
        monkeypatch.setattr("auth._auth_config_override", None)

    def test_websocket_rejects_connection_when_auth_enabled_wrong_key(
        self, monkeypatch
    ):
        """Test that WebSocket rejects connection when auth is enabled and
        wrong key provided."""
        from unittest.mock import MagicMock, patch

        from fastapi.testclient import TestClient

        # Enable auth
        monkeypatch.setattr(
            "auth._auth_config_override", {"enabled": True, "api_key": "secret-key"}
        )

        with patch("torch.hub.load") as mock_hub_load:
            mock_hub_load.return_value = MagicMock()
            from main import app

            client = TestClient(app)

            # WebSocket with wrong API key should be rejected
            with pytest.raises(Exception):  # noqa: B017
                with client.websocket_connect("/ws/test-session?api_key=wrong-key"):
                    pass

        # Clean up
        monkeypatch.setattr("auth._auth_config_override", None)

    def test_websocket_accepts_connection_when_auth_enabled_correct_key(
        self, monkeypatch
    ):
        """Test that WebSocket accepts connection when auth is enabled and
        correct key provided."""
        from unittest.mock import MagicMock, patch

        from fastapi.testclient import TestClient

        # Enable auth
        monkeypatch.setattr(
            "auth._auth_config_override", {"enabled": True, "api_key": "secret-key"}
        )

        with patch("torch.hub.load") as mock_hub_load:
            mock_hub_load.return_value = MagicMock()
            from main import app

            client = TestClient(app)

            # WebSocket with correct API key should be accepted
            # This should NOT raise an exception
            with client.websocket_connect(
                "/ws/test-session?api_key=secret-key"
            ) as websocket:
                # Connection succeeded - send config to complete handshake
                websocket.send_json(
                    {"model": "tiny", "device": "cpu", "language": "en"}
                )
                # We don't need to wait for response, just verify connection was
                # accepted

        # Clean up
        monkeypatch.setattr("auth._auth_config_override", None)

    def test_websocket_accepts_connection_when_auth_disabled(self, monkeypatch):
        """Test that WebSocket accepts connection when auth is disabled."""
        from unittest.mock import MagicMock, patch

        from fastapi.testclient import TestClient

        # Disable auth
        monkeypatch.setattr(
            "auth._auth_config_override", {"enabled": False, "api_key": ""}
        )

        with patch("torch.hub.load") as mock_hub_load:
            mock_hub_load.return_value = MagicMock()
            from main import app

            client = TestClient(app)

            # WebSocket without API key should be accepted when auth is disabled
            with client.websocket_connect("/ws/test-session") as websocket:
                websocket.send_json(
                    {"model": "tiny", "device": "cpu", "language": "en"}
                )
                # Connection succeeded

        # Clean up
        monkeypatch.setattr("auth._auth_config_override", None)
