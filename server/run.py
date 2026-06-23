"""Entrypoint: `python run.py` starts the FastAPI server with settings from .env."""

from __future__ import annotations

import uvicorn

from app.config import settings

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
