from fastapi import HTTPException


# 403
class InvalidAuthenticationSchemeException(HTTPException):
    def __init__(self, detail: str = "Invalid authentication scheme."):
        super().__init__(status_code=403, detail=detail)


class InvalidAPIKeyException(HTTPException):
    def __init__(self, detail: str = "Invalid API key."):
        super().__init__(status_code=403, detail=detail)


# 404
class ModelNotFoundException(HTTPException):
    def __init__(self, detail: str = "Model not found."):
        super().__init__(status_code=404, detail=detail)
