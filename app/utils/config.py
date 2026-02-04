import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Bootstrap settings"""

    app_name: str = "whisperx-api"
    app_version: str = "0.0.0"
    batch_size: int = 16
    api_key: str
    hf_token: str
    timeout_keep_alive: int = 60
    return_char_alignments: bool = False
    interpolate_method: str = "nearest"
    fill_nearest: bool = False

    # Server settings (previously command-line args)
    model: str = "large-v2"
    port: int = 8000
    reload: bool = False
    root_path: str | None = None
    logging_config: str | None = None
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
