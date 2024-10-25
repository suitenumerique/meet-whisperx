from typing import Optional, Union

from fastapi import APIRouter, Security, Annotated

from app.schemas.models import Models, Model
from app.utils.security import check_api_key
from app.utils.args import args


router = APIRouter()

@router.get("/models/{model:path}", tags=["Models"])
@router.get("/models", tags=["Models"])
async def models(
    api_key: Annotated[str, Security(check_api_key)], 
    model: Optional[str] = None
) -> Union[Models, Model]:
    """
    Model API similar to OpenAI's API.
    See https://platform.openai.com/docs/api-reference/models/list for the API specification.
    """

    data = [Model(id=args.model)]

    if model is not None:
        return data[0]

    return Models(data=data)