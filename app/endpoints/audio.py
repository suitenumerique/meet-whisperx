from fastapi import APIRouter, Body, File, Security, UploadFile
from fastapi.responses import PlainTextResponse

from schemas.audio import Transcription, TranscriptionRequest
from utils.lifespan import pipelines
from utils.security import check_api_key

router = APIRouter()


@router.post("/audio/transcriptions")
async def audio_transcriptions(file: UploadFile = File(...), request: TranscriptionRequest = Body(...), api_key=Security(check_api_key)):
    """
    Audio transcriptions API similar to OpenAI's API.
    See https://platform.openai.com/docs/api-reference/audio/create-transcription for the API specification.
    """

    file = await file.read()
    result = pipelines[request.model](
        file, generate_kwargs={"language": request.language, "temperature": request.temperature}, return_timestamps=True
    )

    if request.response_format == "text":
        return PlainTextResponse(result["text"])

    return Transcription(**result)
