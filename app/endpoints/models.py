from typing import Optional, Union
import datetime as dt

from fastapi import APIRouter, Security

from schemas.models import Model, Models
from utils.args import args
from utils.security import check_api_key

router = APIRouter()


@router.get("/models/{model:path}")
@router.get("/models")
async def models(
    model: Optional[str] = None, api_key=Security(check_api_key)
) -> Union[Models, Model]:
    """
    Model API similar to OpenAI's API.
    See https://platform.openai.com/docs/api-reference/models/list for the API specification.
    """

    data = [
        Model(
            object="model",
            id=args.model,
            created=round(dt.datetime.now().timestamp()),
            owned_by="whisper-openai-api",
            type="automatic-speech-recognition",
        )
    ]

    if model is not None:
        return data[0]

    return Models(data=data)
