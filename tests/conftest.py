# tests/conftest.py

import os

import pytest
import pytest_asyncio

BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="session", autouse=True)
def enforce_offline_model_tests():
    """Prevent accidental model/network downloads during tests."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


@pytest.fixture(scope="session")
def base_url():
    """Provides the base URL for the running application."""
    return BASE_URL


# Fixture to provide the FastAPI app for testing
@pytest.fixture(scope="session")
def app():
    """Provides the FastAPI application instance for testing."""
    # Disable model preloading for tests by patching config before import
    import unittest.mock as mock

    # Mock silero-vad loader to avoid loading actual VAD model during tests
    with mock.patch("silero_vad.load_silero_vad") as mock_load_vad:
        mock_load_vad.return_value = mock.MagicMock()

        # Import after patching
        from backend.main import app as fastapi_app

        mock_whisper_model = mock.MagicMock()
        mock_whisper_model.transcribe.return_value = ([], {})

        with (
            mock.patch(
                "backend.main.get_whisper_model",
                new=mock.AsyncMock(return_value=mock_whisper_model),
            ),
            mock.patch(
                "backend.main.get_whisper_model_sync", return_value=mock_whisper_model
            ),
            mock.patch(
                "backend.main.TranscriptionSession.run_pipeline",
                new=mock.AsyncMock(return_value=None),
            ),
        ):
            yield fastapi_app


@pytest_asyncio.fixture
async def async_client(app):
    """Provides an async HTTP client for testing the FastAPI app."""
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sync_test_client(app):
    """Provides a synchronous test client for testing."""
    from starlette.testclient import TestClient

    with TestClient(app) as client:
        yield client
