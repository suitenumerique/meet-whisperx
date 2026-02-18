import os
import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. Mock heavy third-party libraries in sys.modules BEFORE any app import.
#    app modules import whisperx, torch, numpy at the top level.
# ---------------------------------------------------------------------------
_mock_np = MagicMock()
_mock_torch = MagicMock()
_mock_torch.cuda.is_available.return_value = False
_mock_torch.float32 = "float32"

_mock_whisperx = MagicMock()
FAKE_LANGUAGES = {"en": "english", "fr": "french", "cz": "czech"}
FAKE_ALIGN_MODELS_HF = {"en": "WAV2VEC2_ASR_BASE_960H", "fr": "some-fr-model"}
FAKE_ALIGN_MODELS_TORCH = {"en": "WAV2VEC2_ASR_BASE_960H"}

_mock_whisperx.utils.LANGUAGES = FAKE_LANGUAGES
_mock_whisperx.alignment.DEFAULT_ALIGN_MODELS_HF = FAKE_ALIGN_MODELS_HF
_mock_whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH = FAKE_ALIGN_MODELS_TORCH

for mod_name, mock_obj in {
    "numpy": _mock_np,
    "np": _mock_np,
    "torch": _mock_torch,
    "whisperx": _mock_whisperx,
    "whisperx.utils": _mock_whisperx.utils,
    "whisperx.alignment": _mock_whisperx.alignment,
    "whisperx.asr": _mock_whisperx.asr,
    "whisperx.diarize": _mock_whisperx.diarize,
}.items():
    sys.modules.setdefault(mod_name, mock_obj)

# ---------------------------------------------------------------------------
# 2. Set required env vars BEFORE any app module is imported.
#    config.py and security.py evaluate settings at module level.
# ---------------------------------------------------------------------------
TEST_API_KEY = "test-api-key"
TEST_HF_TOKEN = "test-hf-token"

os.environ.setdefault("API_KEY", TEST_API_KEY)
os.environ.setdefault("HF_TOKEN", TEST_HF_TOKEN)

# ---------------------------------------------------------------------------
# 3. Now it is safe to import app modules.
# ---------------------------------------------------------------------------
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
import pytest  # noqa: E402

from endpoints import audio  # noqa: E402
from utils.config import Settings, get_settings  # noqa: E402
from utils.security import check_api_key  # noqa: E402

_FAKE_WORDS = [
    {"word": "Hello", "start": 0.0, "end": 0.7, "score": 0.95},
    {"word": "world.", "start": 0.8, "end": 1.5, "score": 0.92},
]

MOCK_TRANSCRIPTION_RESULT = {
    "segments": [
        {"start": 0.0, "end": 1.5, "text": "Hello world.", "words": _FAKE_WORDS}
    ],
    "word_segments": _FAKE_WORDS,
}

FAKE_AUDIO = MagicMock(name="fake_audio_array")


def _test_settings() -> Settings:
    return Settings(
        api_key=TEST_API_KEY,
        hf_token=TEST_HF_TOKEN,
        transcribe_model="large-v2",
        batch_size=4,
    )


@pytest.fixture()
def client():
    """TestClient with dependency overrides and lifespan disabled."""
    app = FastAPI()
    app.include_router(audio.router, prefix="/v1")

    app.dependency_overrides[get_settings] = _test_settings
    app.dependency_overrides[check_api_key] = lambda: TEST_API_KEY

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def mock_whisperx():
    """Patch whisperx symbols used directly in the endpoint module."""
    with patch.object(
        audio.whisperx, "load_audio", return_value=FAKE_AUDIO
    ) as load_audio:
        yield {"load_audio": load_audio, "fake_audio": FAKE_AUDIO}


@pytest.fixture()
def mock_transcribe():
    """Patch the transcribe service function as imported in the endpoint module."""
    with patch.object(
        audio,
        "transcribe",
        return_value=MOCK_TRANSCRIPTION_RESULT,
    ) as mock:
        yield mock


@pytest.fixture()
def sample_audio_bytes() -> bytes:
    """Minimal bytes to simulate an uploaded audio file."""
    return b"\x00" * 1024
