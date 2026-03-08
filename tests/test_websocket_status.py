"""Hermetic tests for WebSocket status and input validation."""

from unittest.mock import AsyncMock, patch

import pytest
from starlette.websockets import WebSocketDisconnect

pytestmark = pytest.mark.asyncio


def _collect_statuses(websocket, max_messages: int = 4) -> list[dict]:
    """Collect status payloads until socket closes or enough messages are read."""
    statuses = []
    for _ in range(max_messages):
        try:
            payload = websocket.receive_json()
        except WebSocketDisconnect:
            break
        if payload.get("type") == "status":
            statuses.append(payload)
            if payload.get("status") in {"ready", "error"}:
                break
    return statuses


class TestWebSocketStatusMessages:
    """Tests for deterministic status message behavior."""

    def test_cached_model_emits_loading_then_ready(self, sync_test_client):
        with patch("backend.main.is_model_cached", return_value=True):
            with sync_test_client.websocket_connect(
                "/ws/test-status-cached"
            ) as websocket:
                websocket.send_json(
                    {"model": "tiny", "device": "cpu", "language": "en"}
                )
                statuses = _collect_statuses(websocket)

        assert [s["status"] for s in statuses][:2] == ["loading", "ready"]
        assert all("message" in s and "progress" in s for s in statuses)

    def test_uncached_model_emits_downloading_then_ready(self, sync_test_client):
        with patch("backend.main.is_model_cached", return_value=False):
            with sync_test_client.websocket_connect(
                "/ws/test-status-uncached"
            ) as websocket:
                websocket.send_json(
                    {"model": "tiny", "device": "cpu", "language": "en"}
                )
                statuses = _collect_statuses(websocket)

        assert [s["status"] for s in statuses][:2] == ["downloading", "ready"]

    def test_valid_models_emit_ready(self, sync_test_client):
        valid_models = [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
            "distil-large-v3",
        ]
        with patch("backend.main.is_model_cached", return_value=True):
            for model in valid_models:
                with sync_test_client.websocket_connect(
                    f"/ws/test-valid-model-{model}"
                ) as websocket:
                    websocket.send_json(
                        {"model": model, "device": "cpu", "language": "en"}
                    )
                    statuses = _collect_statuses(websocket)
                assert any(s["status"] == "ready" for s in statuses)


class TestWebSocketErrorHandling:
    """Tests for explicit error signaling."""

    def test_invalid_json_returns_error_status(self, sync_test_client):
        with sync_test_client.websocket_connect("/ws/test-invalid-json") as websocket:
            websocket.send_text("not valid json {")
            payload = websocket.receive_json()

        assert payload["type"] == "status"
        assert payload["status"] == "error"
        assert isinstance(payload["message"], str)

    async def test_internal_errors_are_sanitized(self, sync_test_client, monkeypatch):
        monkeypatch.setattr(
            "backend.main.get_whisper_model",
            AsyncMock(side_effect=RuntimeError("sensitive internals")),
        )
        with patch("backend.main.is_model_cached", return_value=True):
            with sync_test_client.websocket_connect(
                "/ws/test-internal-error"
            ) as websocket:
                websocket.send_json(
                    {"model": "tiny", "device": "cpu", "language": "en"}
                )
                statuses = _collect_statuses(websocket, max_messages=3)

        error_status = next(s for s in statuses if s["status"] == "error")
        assert error_status["message"] == "Unexpected server error"
        assert error_status["error_code"] == "WEBSOCKET_INTERNAL_ERROR"
        assert isinstance(error_status["correlation_id"], str)
        assert "sensitive internals" not in str(error_status)


class TestWebSocketConfigValidation:
    """Tests for WebSocket path/session validation."""

    def test_session_id_max_length_enforced(self, sync_test_client):
        with pytest.raises(WebSocketDisconnect) as exc:
            with sync_test_client.websocket_connect(f"/ws/{'a' * 1000}"):
                pass
        assert exc.value.code == 4002

    @pytest.mark.parametrize("session_id", ["bad$id", "bad.id", "bad space"])
    def test_session_id_invalid_characters_rejected(self, sync_test_client, session_id):
        with pytest.raises(WebSocketDisconnect) as exc:
            with sync_test_client.websocket_connect(f"/ws/{session_id}"):
                pass
        assert exc.value.code == 4003
