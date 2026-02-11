from contextlib import asynccontextmanager

from fastapi import FastAPI
import whisperx

from utils.config import get_device, get_dtype, get_settings

pipelines = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()
    torch_dtype = get_dtype()
    settings = get_settings()

    # Downloads weights (cached by default) + load in memory
    pipelines["transcribe_model"] = whisperx.load_model(
        settings.transcribe_model, device, compute_type=torch_dtype
    )
    pipelines["diarize_model"] = whisperx.DiarizationPipeline(
        use_auth_token=settings.hf_token, device=device
    )
    pipelines["align_models"] = {}
    for language in settings.preloaded_align_model_languages:
        pipelines["align_models"][language] = whisperx.load_align_model(
            language_code=language, device=device
        )

    yield

    pipelines.clear()
