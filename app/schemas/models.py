from typing import List, Literal

from openai.types import Model
from pydantic import BaseModel


class Model(Model):
    type: Literal["automatic-speech-recognition"]


class Models(BaseModel):
    object: Literal["list"] = "list"
    data: List[Model]
