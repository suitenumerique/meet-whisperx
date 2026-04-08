"""Conftest for integration tests — no mocking of whisperx/torch/numpy."""

import os

import pytest


@pytest.fixture()
def integration_client():
    """TestClient using the real app with real model loading."""
    from fastapi.testclient import TestClient
    from main import app

    api_key = os.environ.get("API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    with TestClient(app, headers=headers) as c:
        yield c

    app.dependency_overrides.clear()
