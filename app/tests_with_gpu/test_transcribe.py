"""Integration test with real audio and real WhisperX inference.

Requires inference dependencies (torch, whisperx) and HF_TOKEN env var.
Replace the fixture files to match your real audio and model output:
  - fixtures/sample_en_1.ogg           → a real audio clip
  - fixtures/sample_en_1_expected.json  → the expected API response

Run with:
  uv run pytest app/tests_with_gpu -m integration
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("whisperx", reason="whisperx not installed")

FIXTURES = Path(__file__).parent / "fixtures"
ENDPOINT = "/v1/audio/transcriptions"


@pytest.fixture()
def sample_ogg() -> bytes:
    return (FIXTURES / "sample_en_1.ogg").read_bytes()


@pytest.fixture()
def expected_output() -> dict:
    return json.loads((FIXTURES / "sample_en_1_expected.json").read_text())


@pytest.mark.integration
class TestTranscribeIntegration:
    def test_real_audio_output(self, integration_client, sample_ogg, expected_output):
        """Full request with a real audio file produces the expected response."""
        response = integration_client.post(
            ENDPOINT,
            files={"file": ("sample_en_1.ogg", sample_ogg, "audio/ogg")},
        )

        assert response.status_code == 200
        body = response.json()

        assert body["segments"] == expected_output["segments"]
