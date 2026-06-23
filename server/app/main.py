"""FastAPI application entrypoint: lifespan model loading, routes, and mounts."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from strawberry.fastapi import GraphQLRouter

from app.config import settings
from app.graphql_schema import schema
from app.model_loader import load_model
from app.model_loader import state as model_state
from app.sse_routes import router as sse_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vibethinker.main")

CLIENT_DIR = Path(__file__).resolve().parent.parent.parent / "client"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model exactly once when the server starts."""
    logger.info("Starting up: loading model '%s'...", settings.model_source)
    load_model()
    if model_state.is_loaded:
        logger.info("Model ready on device '%s'", model_state.device)
    else:
        logger.error("Model failed to load: %s", model_state.load_error)
    yield
    logger.info("Shutting down.")


app = FastAPI(title="VibeThinker Local Chat", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Logs the full error server-side, but never leaks internals to the client."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error."})


@app.get("/")
async def root() -> dict:
    """Basic liveness/status JSON for the API root."""
    return {
        "service": "vibethinker-local-chat",
        "status": "ok",
        "model_loaded": model_state.is_loaded,
    }


@app.get("/health")
async def health() -> dict:
    """Reports whether the model is loaded, on which device, and its source."""
    return {
        "model_loaded": model_state.is_loaded,
        "device": model_state.device,
        "model_source": model_state.model_source or settings.model_source,
        "loaded_in_4bit": model_state.loaded_in_4bit,
        "load_error": model_state.load_error,
    }


graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")
app.include_router(sse_router)

app.mount("/client", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client")
