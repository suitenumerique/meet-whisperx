import logging

import uvicorn
from fastapi import FastAPI

from endpoints import monitoring, models, audio
from utils.args import args
from utils.config import APP_VERSION, TIMEOUT_KEEP_ALIVE
from utils.lifespan import lifespan


# Setup logging
level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(format="%(levelname)s:%(asctime)s:%(name)s: %(message)s", level=level)


# Setup FastAPI
app = FastAPI(title="Whisper OpenAI API", version=APP_VERSION, licence_info={"name": "MIT License", "identifier": "MIT"}, lifespan=lifespan)

app.include_router(monitoring.router, tags=["Monitoring"])
app.include_router(models.router, tags=["Model"], prefix="/v1")
app.include_router(audio.router, tags=["Audio"], prefix="/v1")


if __name__ == "__main__":
    app.root_path = args.root_path
    uvicorn.run(
        'main:app',
        host="0.0.0.0",
        port=args.port,
        log_level="debug" if args.debug else "info",
        log_config=args.logging_config,
        reload=args.reload,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
