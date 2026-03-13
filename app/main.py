"""
Financial AI Assistant — Main Application Entry Point.

This module initializes the FastAPI application, configures logging,
and includes the API router.
"""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.rag.index_builder import DEFAULT_INDEX_DIR

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Application lifecycle hooks with deployment readiness checks."""
    logger.info("Financial AI Assistant is starting up ...")

    if not os.getenv("FINNHUB_API_KEY", "").strip():
        logger.warning("FINNHUB_API_KEY is not configured; stock queries will fail")

    index_dir = Path(DEFAULT_INDEX_DIR)
    if not index_dir.exists() or not any(index_dir.iterdir()):
        logger.warning("FAISS index directory is missing/empty at %s", index_dir)

    yield

    logger.info("Financial AI Assistant is shutting down ...")


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Financial AI Assistant",
    description=(
        "A modular financial insight assistant powered by RAG, "
        "LlamaIndex, Finnhub, and an LLM reasoning agent."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # lock this down to your domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Include API routes
app.include_router(router)


# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions.

    Args:
        request: The incoming HTTP request.
        exc: The unhandled exception.

    Returns:
        A JSON response with a 500 status code.
    """
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


