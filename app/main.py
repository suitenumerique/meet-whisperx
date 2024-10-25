import argparse
from contextlib import asynccontextmanager
from typing import Annotated
import logging
import uvicorn
from fastapi import FastAPI, Response, Security
from security import check_api_key
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_flash_attn_2_available

from typing import Optional, Union


from app.schemas.models import Model, Models


from app.config import APP_VERSION, TIMEOUT_KEEP_ALIVE

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="openai/whisper-large-v3")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


# Setup logging
logging.basicConfig(format="%(levelname)s:%(asctime)s:%(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
level = logging.DEBUG if args.debug else logging.INFO
logger.setLevel(level)


# Local model
pipe = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model.to(device)

    pipe["model"] = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )

    yield

    pipe.clear()

# Setup FastAPI
app = FastAPI(title="Whisper OpenAI API", version=APP_VERSION, licence_info={"name": "MIT License", "identifier": "MIT"}, lifespan=lifespan)


@app.get("/health", tags=["Monitoring"])
def health(api_key: Annotated[str, Security(check_api_key)]):
    """
    Health check.
    """

    return Response(status_code=200)


@app.get("/models/{model:path}", tags=["Models"])
@app.get("/models", tags=["Models"])
async def models(api_key: Annotated[str, Security(check_api_key)], model: Optional[str] = None) -> Union[Models, Model]:
    """
    Model API similar to OpenAI's API.
    See https://platform.openai.com/docs/api-reference/models/list for the API specification.
    """

    data = [Model(id=args.model)]

    if model is not None:
        return data[0]

    return Models(data=data)


@app.post("/audio/transcriptions", tags=["Audio"])
async def audio_transcriptions(api_key: Annotated[str, Security(check_api_key)], file: UploadFile = File(...)) -> Transcription:
    """
    Audio transcriptions API similar to OpenAI's API.
    See https://platform.openai.com/docs/api-reference/audio/create-transcription for the API specification.
    """

    pass

if __name__ == "__main__":
    level = "DEBUG" if args.debug else "INFO"
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s: %(message)s",
        level=logging.getLevelName(level),
    )

    app.root_path = args.root_path
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.debug else "info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
