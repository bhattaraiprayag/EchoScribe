# tests/test_api.py

import asyncio
import json
import os
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
    new_settings = original_settings.copy()
    new_threshold = round(original_threshold + 0.1, 2)
    if new_threshold > 0.9:  # Stay within valid bounds
        new_threshold = round(original_threshold - 0.1, 2)
    new_settings["vad_parameters"]["prob_threshold"] = new_threshold
    post_response = await async_client.post("/api/settings", json=new_settings)
    assert post_response.status_code == 200
    assert post_response.json()["message"] == "Settings updated successfully"
    get_response_2 = await async_client.get("/api/settings")
    assert get_response_2.status_code == 200
    updated_settings = get_response_2.json()
    assert updated_settings["vad_parameters"]["prob_threshold"] == new_threshold
    # Reset to original
    post_response_reset = await async_client.post("/api/settings", json=original_settings)
    assert post_response_reset.status_code == 200


async def test_websocket_connection(sync_test_client):
    """Tests establishing a WebSocket connection and sending the initial config."""
    # Use starlette's TestClient for WebSocket testing
    with sync_test_client.websocket_connect("/ws/test-session") as websocket:
        websocket.send_json({"model": "tiny", "device": "cpu", "language": "en"})
        # The server shouldn't send any immediate response for just config
        # We just verify connection works


@pytest.mark.skipif(
    not (Path(__file__).parent / "transcribe_test.mp3").exists(),
    reason="Test audio file not found"
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
