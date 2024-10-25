import logging

import uvicorn
from fastapi import FastAPI

from app.endpoints import health, models, audio
from app.utils.args import args
from app.utils.config import APP_VERSION, TIMEOUT_KEEP_ALIVE
from app.utils.lifespan import lifespan


# Setup logging
logging.basicConfig(format="%(levelname)s:%(asctime)s:%(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
level = logging.DEBUG if args.debug else logging.INFO
logger.setLevel(level)


# Setup FastAPI
app = FastAPI(
    title="Whisper OpenAI API", 
    version=APP_VERSION, 
    licence_info={"name": "MIT License", "identifier": "MIT"}, 
    lifespan=lifespan
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(audio.router)


if __name__ == "__main__":
    app.root_path = args.root_path
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="debug" if args.debug else "info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
