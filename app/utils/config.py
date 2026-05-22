from functools import lru_cache
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


class Settings(BaseSettings):
    """Bootstrap settings"""

    api_key: str
    hf_token: str | None = None
    app_name: str = "whisperx-api"
    app_version: str = "0.0.0"

    batch_size: int = 16
    transcribe_model: str = "large-v2"
    diarize_model: str = "pyannote/speaker-diarization-community-1"
    preloaded_align_model_languages: list[str] = ["en", "fr", "nl", "de"]
    timeout_keep_alive: Annotated[int, Field(default=60, description="Correspond to --keep-alive flag in Gunicorn")]  # fmt: off
    timeout_graceful_shutdown: Annotated[int, Field(default=75, description="Correspond to --graceful-timeout flag in Gunicorn")]  # fmt: off

    return_char_alignments: bool = False
    interpolate_method: str = "nearest"
    fill_nearest: bool = False

    port: int = 8000

    workers: int = 1

    # Gunicorn settings
    timeout: Annotated[int, Field(default=120, description="Correspond to --timeout flag in Gunicorn")]  # fmt: off
    worker_connections: Annotated[int, Field(default=1000, description="Correspond to --worker-connections flag in Gunicorn")]  # fmt: off

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
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache
def get_dtype():
    return "float16" if torch.cuda.is_available() else torch.float32
