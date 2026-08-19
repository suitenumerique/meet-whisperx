from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


class Settings(BaseSettings):
    """Bootstrap settings"""

    api_key: str
    hf_token: str
    app_name: str = "whisperx-api"
    app_version: str = "0.0.0"

    batch_size: int = 16
    transcribe_model: str = "large-v2"
    diarize_model: str = "pyannote/speaker-diarization-community-1"
    preloaded_align_model_languages: list[str] = ["en", "fr", "nl", "de"]
    timeout_keep_alive: int = 60

    device: str | None = None
    compute_type: str | None = None
    cpu_threads: int = 4

    return_char_alignments: bool = False
    interpolate_method: str = "nearest"
    fill_nearest: bool = False

    port: int = 8000
    reload: bool = False
    root_path: str | None = None
    logging_config: str | None = "logging-config.yaml"
    debug: bool = False

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings():
    return Settings()


settings = get_settings()


@lru_cache
def get_device():
    """Device running the torch models: alignment and diarization."""
    if settings.device:
        return settings.device

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@lru_cache
def get_asr_device():
    """Device running the CTranslate2 model, which has no Metal backend."""
    return "cuda" if get_device() == "cuda" else "cpu"


@lru_cache
def get_dtype():
    if settings.compute_type:
        return settings.compute_type

    return "float16" if get_asr_device() == "cuda" else "float32"
