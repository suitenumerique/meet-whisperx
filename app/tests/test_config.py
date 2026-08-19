"""Resolution of the inference device and of the compute type."""

import pytest

from utils import config

RESOLVERS = (config.get_device, config.get_asr_device, config.get_dtype)


@pytest.fixture(autouse=True)
def isolated_resolvers(monkeypatch):
    """The resolvers are cached and read the module-level settings."""
    monkeypatch.setattr(config.settings, "device", None)
    monkeypatch.setattr(config.settings, "compute_type", None)
    for resolver in RESOLVERS:
        resolver.cache_clear()
    yield
    for resolver in RESOLVERS:
        resolver.cache_clear()


def _detect(monkeypatch, *, cuda=False, mps=False):
    monkeypatch.setattr(config.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(config.torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, False, "cuda"),
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_device_detection_prefers_cuda_then_mps(monkeypatch, cuda, mps, expected):
    _detect(monkeypatch, cuda=cuda, mps=mps)

    assert config.get_device() == expected


def test_configured_device_wins_over_detection(monkeypatch):
    _detect(monkeypatch, cuda=True)
    monkeypatch.setattr(config.settings, "device", "cpu")

    assert config.get_device() == "cpu"


def test_transcription_never_runs_on_mps(monkeypatch):
    """CTranslate2 has no Metal backend, so the ASR model stays on cpu."""
    _detect(monkeypatch, mps=True)

    assert config.get_device() == "mps"
    assert config.get_asr_device() == "cpu"


@pytest.mark.parametrize(
    ("cuda", "expected"),
    [(True, "float16"), (False, "float32")],
)
def test_compute_type_is_a_ctranslate2_identifier(monkeypatch, cuda, expected):
    """CTranslate2 names its compute types and rejects a torch dtype outright."""
    _detect(monkeypatch, cuda=cuda)

    assert config.get_dtype() == expected


def test_configured_compute_type_wins_over_detection(monkeypatch):
    _detect(monkeypatch, cuda=True)
    monkeypatch.setattr(config.settings, "compute_type", "int8")

    assert config.get_dtype() == "int8"
