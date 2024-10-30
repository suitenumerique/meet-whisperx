from typing import List, Literal

from fastapi import APIRouter, File, Form, Request, Security, UploadFile
from fastapi.responses import PlainTextResponse
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE, LANGUAGES

from schemas.audio import AudioTranscription, AudioTranscriptionVerbose
from utils.args import args
from utils.exceptions import ModelNotFoundException
from utils.lifespan import pipelines
from utils.security import check_api_key


router = APIRouter()

SUPPORTED_LANGUAGES = set(list(LANGUAGES.keys()) + list(TO_LANGUAGE_CODE.keys()))


@router.post("/audio/transcriptions")
async def audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(args.model),
    language: str = Form("en"),
    prompt: str = Form(None), # TODO: implement
    response_format: Literal["text", "json"] = Form("json"),
    temperature: float = Form(0),
    timestamp_granularities: List[str] = Form(alias="timestamp_granularities[]", default=["segment"]), # TODO: implement
    api_key=Security(check_api_key)
) -> AudioTranscription | AudioTranscriptionVerbose:
    """
    Audio transcriptions API similar to OpenAI's API.
    See https://platform.openai.com/docs/api-reference/audio/create-transcription for the API specification.
    """
    
    if model != args.model:
        raise ModelNotFoundException()
    
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Language {language} not supported")

    file = await file.read()
    result = pipelines[model](
        file, generate_kwargs={"language": language, "temperature": temperature}, return_timestamps=True
    )

    if response_format == "text":
        return PlainTextResponse(result["text"])

    return AudioTranscription(**result)
