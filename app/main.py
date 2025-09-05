import uvicorn
from fastapi import FastAPI, Depends

from typing import Annotated

from endpoints import monitoring, models, audio
from utils.args import args
from utils.lifespan import lifespan
from utils.config import get_settings, Settings, settings


# Setup FastAPI
app = FastAPI(
    title="Whisper OpenAI API",
    version=settings.app_version,
    licence_info={"name": "MIT License", "identifier": "MIT"},
    lifespan=lifespan,
)

app.include_router(monitoring.router, tags=["Monitoring"])
app.include_router(models.router, tags=["Model"], prefix="/v1")
app.include_router(audio.router, tags=["Audio"], prefix="/v1")


@app.get("/info")
async def info(settings: Annotated[Settings, Depends(get_settings)]):
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "batch_size": settings.batch_size,
    }


if __name__ == "__main__":
    app.root_path = args.root_path
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=args.port,
        log_level="debug" if args.debug else "info",
        log_config=args.logging_config,
        reload=args.reload,
        timeout_keep_alive=settings.timeout_keep_alive,
    )
