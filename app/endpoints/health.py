from fastapi import APIRouter, Security, Annotated, Response

from app.utils.security import check_api_key


router = APIRouter()

@router.get("/health", tags=["Monitoring"])
def health(api_key: Annotated[str, Security(check_api_key)]):
    """
    Health check.
    """

    return Response(status_code=200)
