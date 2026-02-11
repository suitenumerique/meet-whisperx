from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI
import whisperx
from whisperx.asr import FasterWhisperPipeline
from whisperx.diarize import DiarizationPipeline

from utils.config import get_device, get_dtype, get_settings


@dataclass
class Pipelines:
    transcribe_model: FasterWhisperPipeline | None = None
    diarize_model: DiarizationPipeline | None = None
    align_models: dict[str, tuple[Any, dict]] = field(default_factory=dict)


pipelines = Pipelines()


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()
    torch_dtype = get_dtype()
    settings = get_settings()

    # Downloads weights (cached by default) + load in memory
    pipelines.transcribe_model = whisperx.load_model(
        settings.transcribe_model, device, compute_type=torch_dtype
    )
    pipelines.diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=settings.hf_token, device=device
    )
    for language in settings.preloaded_align_model_languages:
        pipelines.align_models[language] = whisperx.load_align_model(
            language_code=language, device=device
        )

    yield

    pipelines.transcribe_model = None
    pipelines.diarize_model = None
    pipelines.align_models.clear()
