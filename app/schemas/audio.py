from typing import Literal, Optional
import json

from openai.types.audio import Transcription
from pydantic import BaseModel, Field, field_validator, model_validator
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE, LANGUAGES

from utils.args import args
from utils.exceptions import ModelNotFoundException


class Transcription(Transcription):
    pass


class TranscriptionRequest(BaseModel):
    model: str = Field(default=args.model)
    language: Literal[*LANGUAGES.keys(), *TO_LANGUAGE_CODE.keys()] = Field(default="en")
    response_format: Literal["text", "json"] = Field(default="json")
    temperature: Optional[float] = Field(default=None, ge=0, le=1)

    @field_validator("model")
    @classmethod
    def validate_model(cls, model):
        if model != args.model:
            raise ModelNotFoundException()
        return model

    @model_validator(mode="before")
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


class Transcription(BaseModel):
    text: str
