from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from app.utils.args import args


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