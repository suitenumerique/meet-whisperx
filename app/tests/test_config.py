"""Resolution of the compute type handed to CTranslate2."""

import pytest

from utils import config


@pytest.fixture(autouse=True)
def clear_resolver_caches():
    """The resolvers are cached, so every case needs a fresh answer."""
    config.get_dtype.cache_clear()
    yield
    config.get_dtype.cache_clear()


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    [(True, "float16"), (False, "float32")],
)
def test_compute_type_is_a_ctranslate2_identifier(
    monkeypatch, cuda_available, expected
):
    """CTranslate2 names its compute types and rejects a torch dtype outright."""
    monkeypatch.setattr(config.torch.cuda, "is_available", lambda: cuda_available)

    assert config.get_dtype() == expected
