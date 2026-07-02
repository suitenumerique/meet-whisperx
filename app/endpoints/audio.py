import asyncio
import logging
import os
import tempfile
import time
from functools import partial
from typing import Annotated, Optional

import numpy as np
import tiktoken
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Security,
    UploadFile,
)
from fastapi.responses import PlainTextResponse
from services.transcription import transcribe
from utils.lifespan import gpu_executor
import whisperx

from schemas.audio import AudioTranscription, InputTokenDetails, Segment, Usage
from utils.config import Settings, get_settings
from utils.exceptions import ModelNotFoundException
from utils.security import check_api_key

logger = logging.getLogger("api")

router = APIRouter()

WHISPERX_SAMPLE_RATE = 16_000
AUDIO_TOKENS_PER_SECOND = 10


SUPPORTED_RESPONSE_FORMATS = {"json", "text", "verbose_json", "diarized_json", "srt", "vtt"}


def _format_timestamp(seconds: float, separator: str) -> str:
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def _format_srt(segments: list[dict]) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp(seg["start"], ",")
        end = _format_timestamp(seg["end"], ",")
        blocks.append(f"{i}\n{start} --> {end}\n{seg['text'].strip()}")
    return "\n\n".join(blocks) + "\n"


def _format_vtt(segments: list[dict]) -> str:
    blocks = ["WEBVTT"]
    for seg in segments:
        start = _format_timestamp(seg["start"], ".")
        end = _format_timestamp(seg["end"], ".")
        blocks.append(f"{start} --> {end}\n{seg['text'].strip()}")
    return "\n\n".join(blocks) + "\n"


def _build_response(
    result: dict, audio: np.ndarray, is_diarize: bool, is_verbose: bool
) -> AudioTranscription:
    raw_segments = result.get("segments", [])
    text = "".join(seg["text"] for seg in raw_segments)

    audio_tokens = round(len(audio) / WHISPERX_SAMPLE_RATE * AUDIO_TOKENS_PER_SECOND)
    output_tokens = len(tiktoken.get_encoding("o200k_base").encode(text))

    segments = None
    if is_diarize or is_verbose:
        segments = [
            Segment(
                id=i,
                text=seg["text"],
                start=seg["start"],
                end=seg["end"],
                speaker=seg.get("speaker"),
            )
            for i, seg in enumerate(raw_segments)
        ]

    return AudioTranscription(
        task="transcribe" if is_verbose else None,
        language=result.get("language") if is_verbose else None,
        duration=len(audio) / WHISPERX_SAMPLE_RATE if is_verbose else None,
        text=text,
        segments=segments,
        usage=Usage(
            input_tokens=audio_tokens,
            input_token_details=InputTokenDetails(audio_tokens=audio_tokens),
            output_tokens=output_tokens,
            total_tokens=audio_tokens + output_tokens,
        ),
    )


@router.post("/audio/transcriptions", dependencies=[Security(check_api_key)])
async def audio_transcriptions(
    settings: Annotated[Settings, Depends(get_settings)],
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
) -> AudioTranscription:
    """
    Audio transcription API compatible with the OpenAI transcription response format.
    Supported response_format values: "json" (default), "text", "verbose_json", "diarized_json", "srt", "vtt".
    """
    logger.info("Request received. Transcribe model: %s, language: %s", model, language)

    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{response_format}'. Must be one of: {sorted(SUPPORTED_RESPONSE_FORMATS)}.",
        )

    is_diarize = response_format == "diarized_json"
    is_verbose = response_format == "verbose_json"
    is_text = response_format == "text"
    is_srt = response_format == "srt"
    is_vtt = response_format == "vtt"

    if language is not None and language not in whisperx.utils.LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{language}' for transcription.",
        )
    if is_diarize and language is not None and language not in (
        whisperx.alignment.DEFAULT_ALIGN_MODELS_HF
        | whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH
    ):
        raise HTTPException(
            status_code=400, detail=f"Unsupported language '{language}' for alignment."
        )

    if model is None:
        model = settings.transcribe_model
    if model != settings.transcribe_model:
        raise ModelNotFoundException()

    logger.info("Reading file …")
    reading_start = time.perf_counter()
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as temp_file:
        temp_file_path = temp_file.name
        content = await file.read()
        temp_file.write(content)

    reading_time = time.perf_counter() - reading_start
    logger.info("Reading time: %.3fs", reading_time)

    logger.info("Loading audio file to whisper…")
    audio = whisperx.load_audio(temp_file_path)
    os.remove(temp_file_path)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        gpu_executor, partial(transcribe, audio, settings, language, is_diarize=is_diarize)
    )

    segments = result.get("segments", [])

    if is_text:
        return PlainTextResponse(content="".join(seg["text"] for seg in segments))
    if is_srt:
        return PlainTextResponse(content=_format_srt(segments), media_type="application/x-subrip")
    if is_vtt:
        return PlainTextResponse(content=_format_vtt(segments), media_type="text/vtt")

    return _build_response(result, audio, is_diarize=is_diarize, is_verbose=is_verbose)
