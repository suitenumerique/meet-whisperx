from fastapi import APIRouter, File, Form, Request, Security, UploadFile, Depends

import whisperx

from schemas.audio import AudioTranscription, AudioTranscriptionVerbose
from utils.args import args
from utils.exceptions import ModelNotFoundException
from utils.lifespan import pipelines
from utils.security import check_api_key
from utils.config import get_settings, Settings, get_device
from typing import Annotated

import tempfile
import os


router = APIRouter()


@router.post("/audio/transcriptions")
async def audio_transcriptions(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    file: UploadFile = File(...),
    model: str = Form(args.model),
    api_key=Security(check_api_key),
) -> AudioTranscription | AudioTranscriptionVerbose:
    """
    Audio transcriptions API similar to OpenAI's API.
    See https://platform.openai.com/docs/api-reference/audio/create-transcription for the API specification.
    """

    if model != args.model:
        raise ModelNotFoundException()

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file_path = temp_file.name
        content = await file.read()
        temp_file.write(content)

    audio = whisperx.load_audio(temp_file_path)
    os.remove(temp_file_path)

    result = pipelines[args.model].transcribe(audio, batch_size=settings.batch_size)

    device = get_device()
    model_alignment, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(
        result["segments"],
        model_alignment,
        metadata,
        audio,
        device,
        interpolate_method=settings.interpolate_method,
        return_char_alignments=settings.return_char_alignments,
    )

    diarize_segments = pipelines["diarize_model"](audio)

    result = whisperx.assign_word_speakers(diarize_segments, result, fill_nearest=settings.fill_nearest)

    return AudioTranscription(**result)
