import datetime as dt
from typing import Annotated, Optional, Union

from fastapi import APIRouter, Depends, Security

from schemas.models import Model, Models
from utils.config import Settings, get_settings
from utils.security import check_api_key

router = APIRouter()


@router.get("/models/{model:path}")
@router.get("/models")
async def models(
    settings: Annotated[Settings, Depends(get_settings)],
    model: Optional[str] = None,
    api_key=Security(check_api_key),
) -> Union[Models, Model]:
    """
    Model API similar to OpenAI's API.
    See https://platform.openai.com/docs/api-reference/models/list for the API specification.
    """

    data = [
        Model(
            object="model",
            id=settings.transcribe_model,
            created=round(dt.datetime.now().timestamp()),
            owned_by="whisper-openai-api",
            type="automatic-speech-recognition",
        )
    ]

    if model is not None:
        return data[0]

    return Models(data=data)
