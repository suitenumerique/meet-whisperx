from contextlib import asynccontextmanager

from fastapi import FastAPI
import whisperx

from utils.args import args
from utils.config import get_device, get_dtype, get_settings

pipelines = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()
    torch_dtype = get_dtype()

    settings = get_settings()

    pipelines[args.model] = whisperx.load_model(
        args.model, device, compute_type=torch_dtype
    )
    pipelines["diarize_model"] = whisperx.DiarizationPipeline(
        use_auth_token=settings.hf_token, device=device
    )

    yield

    pipelines.clear()
