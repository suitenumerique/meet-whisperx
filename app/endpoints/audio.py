from typing import Union

from fastapi import APIRouter, Security, Annotated, UploadFile, File

from app.schemas.audio import AudioTranscription, AudioTranscriptionVerbose
from app.utils.security import check_api_key
from app.utils.lifespan import pipe


router = APIRouter()

@router.post("/audio/transcriptions", tags=["Audio"])
async def audio_transcriptions(
    api_key: Annotated[str, Security(check_api_key)], 
    file: UploadFile = File(...)
) -> Union[AudioTranscription, AudioTranscriptionVerbose]:
    """
    Audio transcriptions API similar to OpenAI's API.
    See https://platform.openai.com/docs/api-reference/audio/create-transcription for the API specification.
    """
    response = pipe["model"](file,  generate_kwargs={"language": "fr", "temperature": 0.9}, return_timestamps=True)  

    return AudioTranscriptionVerbose(**response)