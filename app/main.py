from typing import Annotated

from fastapi import Depends, FastAPI
import uvicorn

from endpoints import audio, models, monitoring
from utils.config import Settings, get_settings, settings
from utils.lifespan import lifespan

# Setup FastAPI
app = FastAPI(
    title="LaSuite Meet WhisperX",
    version=settings.app_version,
    license_info={"name": "MIT License", "identifier": "MIT"},
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
    app.root_path = settings.root_path
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        log_level="debug" if settings.debug else "info",
        log_config=settings.logging_config,
        reload=settings.reload,
        timeout_keep_alive=settings.timeout_keep_alive,
    )
