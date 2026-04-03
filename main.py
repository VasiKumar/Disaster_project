from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes.api import router as api_router
from app.routes.web import router as web_router
from app.services.video_processor import video_processor

if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

app = FastAPI(title=settings.app_name)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(web_router)
app.include_router(api_router)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": settings.app_name}


@app.on_event("shutdown")
def shutdown_cleanup() -> None:
    # Force camera worker shutdown so Ctrl+C exits quickly.
    video_processor.stop()


if __name__ == "__main__":
    import uvicorn

    # On Windows, uvicorn reload can delay Ctrl+C because of watcher subprocesses.
    enable_reload = settings.debug and os.name != "nt"
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=enable_reload,
        timeout_graceful_shutdown=3,
    )
