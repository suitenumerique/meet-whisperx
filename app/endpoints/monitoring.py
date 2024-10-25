from typing import Annotated

from fastapi import APIRouter, Security, Response

from utils.security import check_api_key


router = APIRouter()


@router.get("/health")
def health(api_key: Annotated[str, Security(check_api_key)]):
    """
    Health check.
    """

    return Response(status_code=200)
