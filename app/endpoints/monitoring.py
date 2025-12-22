from fastapi import APIRouter, Response


router = APIRouter()


@router.get("/health")
def health():
    """
    Health check.
    """

    return Response(status_code=200)
