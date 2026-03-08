# tests/test_api.py

import io
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio
TEST_DIR = Path(__file__).parent


async def test_api_health(async_client):
    """Tests if the API root endpoint responds (frontend may or may not be mounted)."""
    response = await async_client.get("/api/config")
    # Test that the API is responsive - /api/config should always work
    assert response.status_code == 200


async def test_get_config(async_client):
    """Tests the /api/config endpoint."""
    response = await async_client.get("/api/config")
    assert response.status_code == 200
    config = response.json()
    assert "devices" in config
    assert "models" in config
    assert "languages" in config
    assert isinstance(config["devices"], list)
    assert isinstance(config["models"], list)
    assert isinstance(config["languages"], dict)


async def test_settings_flow(async_client):
    """Tests getting, updating, and resetting settings."""
    get_response_1 = await async_client.get("/api/settings")
    assert get_response_1.status_code == 200
    original_settings = get_response_1.json()
    original_threshold = original_settings["vad_parameters"]["prob_threshold"]
    new_threshold = round(original_threshold + 0.1, 2)
    if new_threshold > 0.9:  # Stay within valid bounds
        new_threshold = round(original_threshold - 0.1, 2)
    # Only send the fields that need to be updated (SettingsUpdate has extra="forbid")
    new_settings = {"vad_parameters": {"prob_threshold": new_threshold}}
    post_response = await async_client.post("/api/settings", json=new_settings)
    assert post_response.status_code == 200
    assert post_response.json()["message"] == "Settings updated successfully"
    get_response_2 = await async_client.get("/api/settings")
    assert get_response_2.status_code == 200
    updated_settings = get_response_2.json()
    assert updated_settings["vad_parameters"]["prob_threshold"] == new_threshold
    # Reset to original (only send vad_parameters)
    reset_settings = {"vad_parameters": {"prob_threshold": original_threshold}}
    post_response_reset = await async_client.post("/api/settings", json=reset_settings)
    assert post_response_reset.status_code == 200


async def test_get_settings_requires_auth_when_enabled(async_client, monkeypatch):
    """GET /api/settings should require API key when auth is enabled."""
    monkeypatch.setattr(
        "backend.auth._auth_config_override", {"enabled": True, "api_key": "secret-key"}
    )
    try:
        response = await async_client.get("/api/settings")
        assert response.status_code == 401
        assert "Invalid or missing API key" in response.json()["detail"]
    finally:
        monkeypatch.setattr("backend.auth._auth_config_override", None)


async def test_get_settings_redacts_api_key(async_client, monkeypatch):
    """GET /api/settings should never return raw API key values."""
    monkeypatch.setattr(
        "backend.auth._auth_config_override", {"enabled": True, "api_key": "secret-key"}
    )
    monkeypatch.setattr(
        "backend.main.get_config",
        lambda: {
            "auth": {"enabled": True, "api_key": "secret-key"},
            "vad_parameters": {"prob_threshold": 0.6},
        },
    )
    try:
        response = await async_client.get(
            "/api/settings", headers={"X-API-Key": "secret-key"}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["auth"]["enabled"] is True
        assert payload["auth"]["api_key"] == "***REDACTED***"
    finally:
        monkeypatch.setattr("backend.auth._auth_config_override", None)


async def test_set_settings_returns_error_when_persistence_fails(
    async_client, monkeypatch
):
    """POST /api/settings should propagate persistence failures."""
    from backend.config_manager import ConfigPersistenceError

    monkeypatch.setattr(
        "backend.main.save_config",
        lambda config: (_ for _ in ()).throw(ConfigPersistenceError("disk failure")),
    )
    response = await async_client.post(
        "/api/settings", json={"vad_parameters": {"prob_threshold": 0.6}}
    )
    assert response.status_code == 500
    payload = response.json()
    assert payload["error"]["code"] == "CONFIG_PERSISTENCE_ERROR"
    assert payload["error"]["message"] == "Failed to persist settings update"
    assert isinstance(payload["error"]["correlation_id"], str)
    assert "disk failure" not in str(payload)


async def test_ws_auth_token_requires_api_key_when_auth_enabled(
    async_client, monkeypatch
):
    """WS auth token endpoint should enforce API key when auth is enabled."""
    monkeypatch.setattr(
        "backend.auth._auth_config_override", {"enabled": True, "api_key": "secret-key"}
    )
    try:
        response = await async_client.post("/api/ws-auth-token")
        assert response.status_code == 401
    finally:
        monkeypatch.setattr("backend.auth._auth_config_override", None)


async def test_ws_auth_token_issued_with_valid_api_key(async_client, monkeypatch):
    """WS auth token endpoint should return short-lived token for valid key."""
    monkeypatch.setattr(
        "backend.auth._auth_config_override", {"enabled": True, "api_key": "secret-key"}
    )
    try:
        response = await async_client.post(
            "/api/ws-auth-token", headers={"X-API-Key": "secret-key"}
        )
        assert response.status_code == 200
        payload = response.json()
        assert isinstance(payload.get("token"), str)
        assert payload.get("expires_in_seconds", 0) > 0
    finally:
        monkeypatch.setattr("backend.auth._auth_config_override", None)


async def test_websocket_connection(sync_test_client):
    """Tests establishing a WebSocket connection and sending the initial config."""
    # Use starlette's TestClient for WebSocket testing
    with sync_test_client.websocket_connect("/ws/test-session") as websocket:
        websocket.send_json({"model": "tiny", "device": "cpu", "language": "en"})
        # The server shouldn't send any immediate response for just config
        # We just verify connection works


@pytest.mark.skipif(
    not (Path(__file__).parent / "transcribe_test.mp3").exists(),
    reason="Test audio file not found",
)
async def test_file_transcription(async_client):
    """Tests the file upload endpoint (job submission only, not full transcription)."""
    audio_file_path = TEST_DIR / "transcribe_test.mp3"

    with open(audio_file_path, "rb") as f:
        files = {"file": (audio_file_path.name, f, "audio/mpeg")}
        data = {"model": "tiny", "language": "en", "device": "cpu"}

        post_response = await async_client.post(
            "/api/transcribe", files=files, data=data
        )

        # Just test that the endpoint accepts the file and returns a job_id
        assert post_response.status_code == 200
        response_json = post_response.json()
        assert "job_id" in response_json


async def test_model_status_endpoint(async_client):
    """Tests the /api/model/status endpoint."""
    response = await async_client.get(
        "/api/model/status", params={"model": "tiny", "device": "cpu"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "cached" in data
    assert "model" in data
    assert "device" in data
    assert data["model"] == "tiny"
    assert data["device"] == "cpu"


async def test_model_status_unknown_model(async_client):
    """Tests /api/model/status with unknown model."""
    response = await async_client.get(
        "/api/model/status", params={"model": "nonexistent-xyz", "device": "cpu"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is False
    assert "error" in data


async def test_model_status_includes_repo_id(async_client):
    """Tests that /api/model/status includes repo_id for known models."""
    response = await async_client.get(
        "/api/model/status", params={"model": "base", "device": "cuda"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "repo_id" in data
    assert "Systran" in data["repo_id"]


async def test_config_exposes_upload_limits(async_client):
    """Tests /api/config includes effective upload size limits."""
    response = await async_client.get("/api/config")
    assert response.status_code == 200
    config = response.json()
    assert "upload_parameters" in config
    assert config["upload_parameters"]["max_file_size_mb"] > 0


async def test_oversized_upload_returns_413(async_client, monkeypatch):
    """Uploads larger than effective limit should return HTTP 413."""
    monkeypatch.setattr("backend.main.get_max_file_size_bytes", lambda config=None: 64)
    files = {"file": ("oversized.mp3", io.BytesIO(b"x" * 128), "audio/mpeg")}
    data = {"model": "tiny", "language": "en", "device": "cpu"}
    response = await async_client.post("/api/transcribe", files=files, data=data)
    assert response.status_code == 413
    assert "File too large." in response.json()["detail"]


async def test_config_models_order(async_client):
    """Tests that /api/config returns models in correct order (ascending size)."""
    response = await async_client.get("/api/config")
    assert response.status_code == 200
    config = response.json()

    models = config["models"]
    # Verify large-v3 is last (largest model)
    assert models[-1] == "large-v3"
    # Verify tiny is first (smallest model)
    assert models[0] == "tiny"
    # Verify distil-large-v3 comes before large-v3
    assert models.index("distil-large-v3") < models.index("large-v3")


async def test_cors_headers_present(async_client):
    """Tests that CORS headers are present in responses."""
    # Simulate cross-origin request
    response = await async_client.options(
        "/api/config",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    # CORS preflight should return 200
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


async def test_security_headers_present(async_client):
    """Responses should include baseline security headers."""
    response = await async_client.get("/api/config")
    assert response.status_code == 200
    assert "content-security-policy" in response.headers
    assert "x-content-type-options" in response.headers
    assert "referrer-policy" in response.headers
    csp = response.headers["content-security-policy"]
    assert "object-src 'none'" in csp
