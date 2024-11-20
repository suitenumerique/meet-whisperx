from typing import List, Literal

import logging
import time

from fastapi import APIRouter, File, Form, Request, Security, UploadFile
from fastapi.responses import PlainTextResponse
from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE

from schemas.audio import AudioTranscription, AudioTranscriptionVerbose
from utils.args import args
from utils.exceptions import ModelNotFoundException
from utils.lifespan import pipelines
from utils.security import check_api_key

router = APIRouter()

SUPPORTED_LANGUAGES = set(list(LANGUAGES.keys()) + list(TO_LANGUAGE_CODE.keys()))

logger = logging.getLogger("api")

@router.post("/audio/transcriptions")
async def audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(args.model),
    language: Literal[tuple(SUPPORTED_LANGUAGES)] = Form("en"),
    prompt: str = Form(None),  # TODO: implement
    response_format: Literal["text", "json"] = Form("json"),  # @TODO: implement stt and verbose_json
    temperature: float = Form(0),
    timestamp_granularities: List[str] = Form(alias="timestamp_granularities[]", default=["segment"]),  # @TODO: implement
    api_key=Security(check_api_key),
) -> AudioTranscription | AudioTranscriptionVerbose:
    """
    Audio transcriptions API similar to OpenAI's API.
    See https://platform.openai.com/docs/api-reference/audio/create-transcription for the API specification.
    """

    if model != args.model:
        raise ModelNotFoundException()

    file = await file.read()

    inference_start = time.perf_counter()
    result = pipelines[model](
        file,
        generate_kwargs={
            "forced_decoder_ids": None,
            "input_features": True,
            "language": language,
            "temperature": temperature,
        },
        return_timestamps=True,
    )
    inference_time = time.perf_counter() - inference_start

    logger.info("Model inference time: %.3fs", inference_time)
    logger.info("Audio length: %s bytes", len(file))

    if response_format == "text":
        return PlainTextResponse(result["text"])

    return AudioTranscription(**result)
