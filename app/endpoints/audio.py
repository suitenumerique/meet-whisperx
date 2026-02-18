import logging
import os
import tempfile
import time
from typing import Annotated, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Security,
    UploadFile,
)
from services.transcription import transcribe
import whisperx

from schemas.audio import AudioTranscription, AudioTranscriptionVerbose
from utils.config import Settings, get_settings
from utils.exceptions import ModelNotFoundException
from utils.security import check_api_key

logger = logging.getLogger("api")

router = APIRouter()


@router.post("/audio/transcriptions")
async def audio_transcriptions(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    api_key=Security(check_api_key),
    language: Optional[str] = Form(None),
) -> AudioTranscription | AudioTranscriptionVerbose:
    """
    Audio transcription API (custom implementation).

    /!\ Note: This endpoint is **not** OpenAI API compatible.
    The response format does not follow the OpenAI specification.
    """
    logger.info("Request received. transcribe model: %s, language: %s", model, language)

    if language is not None and language not in whisperx.utils.LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{language}' for transcription.",
        )
    if language is not None and language not in (
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

    result = transcribe(audio, settings, language)

    return AudioTranscription(**result)
