# tests/test_api.py
import pytest
import httpx
import asyncio
import websockets
import os
import json
from pathlib import Path

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

TEST_DIR = Path(__file__).parent

async def test_api_health(base_url):
    """Tests if the API is running and the root serves HTML."""
    async with httpx.AsyncClient() as client:
        response = await client.get(base_url)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

async def test_get_config(base_url):
    """Tests the /api/config endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/api/config")
        assert response.status_code == 200
        config = response.json()
        assert "devices" in config
        assert "models" in config
        assert "languages" in config
        assert isinstance(config["devices"], list)
        assert isinstance(config["models"], list)
        assert isinstance(config["languages"], dict)

async def test_settings_flow(base_url):
    """Tests getting, updating, and resetting settings."""
    async with httpx.AsyncClient() as client:
        # 1. Get original settings
        get_response_1 = await client.get(f"{base_url}/api/settings")
        assert get_response_1.status_code == 200
        original_settings = get_response_1.json()
        original_threshold = original_settings["vad_parameters"]["prob_threshold"]

        # 2. Update a setting
        new_settings = original_settings.copy()
        new_threshold = round(original_threshold + 0.1, 2)
        if new_threshold > 1.0: new_threshold = round(original_threshold - 0.1, 2)
        new_settings["vad_parameters"]["prob_threshold"] = new_threshold
        
        post_response = await client.post(f"{base_url}/api/settings", json=new_settings)
        assert post_response.status_code == 200
        assert post_response.json()["message"] == "Settings updated successfully"

        # 3. Verify the updated setting
        get_response_2 = await client.get(f"{base_url}/api/settings")
        assert get_response_2.status_code == 200
        updated_settings = get_response_2.json()
        assert updated_settings["vad_parameters"]["prob_threshold"] == new_threshold

        # 4. Reset to original settings to clean up
        post_response_reset = await client.post(f"{base_url}/api/settings", json=original_settings)
        assert post_response_reset.status_code == 200

async def test_websocket_connection(base_url):
    """Tests establishing a WebSocket connection and sending the initial config."""
    ws_url = base_url.replace("http", "ws")
    try:
        async with websockets.connect(f"{ws_url}/ws/test-session") as websocket:
            # The connection is implicitly checked by the context manager.
            # If it fails, an exception is raised.
            await websocket.send(json.dumps({
                "model": "tiny",
                "device": "cpu",
                "language": "en"
            }))
            # If the server accepts it and doesn't close the connection, it's a success
            # We can wait for a moment to see if any error message comes
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                # We don't expect any message back immediately, but if one comes, fail
                pytest.fail(f"Received unexpected message from server: {message}")
            except asyncio.TimeoutError:
                # This is expected, the server is waiting for audio
                pass
    except ConnectionRefusedError:
        pytest.fail("WebSocket connection refused. Is the server running?")

async def test_file_transcription(base_url):
    """Tests the entire file transcription workflow."""
    audio_file_path = TEST_DIR / "transcribe_test.mp3"
    transcript_file_path = TEST_DIR / "transcribe_test.txt"

    assert audio_file_path.exists(), "Test audio file (transcribe_test.mp3) not found."
    assert transcript_file_path.exists(), "Expected transcript file (transcribe_test.txt) not found."

    with open(audio_file_path, "rb") as f:
        files = {"file": (audio_file_path.name, f, "audio/mpeg")}
        # Use the 'base' model for better accuracy in tests
        data = {"model": "base", "language": "en", "device": "cuda"}
        
        async with httpx.AsyncClient(timeout=60) as client:
            # 1. Upload file and start transcription
            post_response = await client.post(f"{base_url}/api/transcribe", files=files, data=data)
            assert post_response.status_code == 200
            job_id = post_response.json().get("job_id")
            assert job_id

            # 2. Poll for status until completed
            while True:
                status_response = await client.get(f"{base_url}/api/transcribe/status/{job_id}")
                assert status_response.status_code == 200
                status_data = status_response.json()
                
                if status_data["status"] == "completed":
                    # 3. Verify the result
                    expected_transcript = transcript_file_path.read_text().strip()
                    actual_transcript = status_data["result"].strip()
                    # Normalize for comparison
                    expected_normalized = "".join(filter(str.isalnum, expected_transcript)).lower()
                    actual_normalized = "".join(filter(str.isalnum, actual_transcript)).lower()
                    assert actual_normalized == expected_normalized
                    break
                elif status_data["status"] == "error":
                    pytest.fail(f"Transcription job failed with error: {status_data['result']}")
                
                await asyncio.sleep(2) # Wait before polling again
