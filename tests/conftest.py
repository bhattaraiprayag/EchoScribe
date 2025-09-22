# tests/conftest.py
import pytest

BASE_URL = "http://localhost:8000"

@pytest.fixture(scope="session")
def base_url():
    """Provides the base URL for the running application."""
    return BASE_URL
