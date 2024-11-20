from contextlib import asynccontextmanager

import logging
from fastapi import FastAPI
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_flash_attn_2_available

from utils.args import args

logger = logging.getLogger("api")

pipelines = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info("Device: %s", device)
    logger.info("Torch_dtype: %s", torch_dtype)
    logger.info("Flash attention 2 available: %s", is_flash_attn_2_available())

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        force_download=args.force_download,
        token=args.hf_token,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model.to(device)

    logger.info("Tokenizer: %s", processor.tokenizer)
    logger.info("Feature extractor: %s", processor.feature_extractor)

    pipelines[args.model] = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )

    yield

    pipelines.clear()
