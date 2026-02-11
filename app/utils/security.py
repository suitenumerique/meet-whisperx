from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from utils.config import settings
from utils.exceptions import (
    InvalidAPIKeyException,
    InvalidAuthenticationSchemeException,
)

auth_scheme = HTTPBearer(scheme_name="API key")

if not settings.api_key:

    def check_api_key():
        # if No API key is set, so we don't check for it.
        pass

else:

    def check_api_key(
        api_key: Annotated[HTTPAuthorizationCredentials, Depends(auth_scheme)],
    ):
        if api_key.scheme != "Bearer":
            raise InvalidAuthenticationSchemeException()
        if api_key.credentials != settings.api_key:
            raise InvalidAPIKeyException()

        return api_key.credentials
