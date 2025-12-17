# tests/conftest.py

import os
import sys

import pytest
import pytest_asyncio


# Add backend directory to Python path for imports
BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

BASE_URL = "http://localhost:8000"

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

    # Mock torch.hub.load to avoid loading actual VAD model during tests
    with mock.patch('torch.hub.load') as mock_hub_load:
        mock_hub_load.return_value = mock.MagicMock()

        # Import after patching
        from main import app as fastapi_app
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
