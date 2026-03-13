"""
Financial AI Assistant — Main Application Entry Point.

This module initializes the FastAPI application, configures logging,
and includes the API router.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Financial AI Assistant",
    description=(
        "A modular financial insight assistant powered by RAG, "
        "LlamaIndex, Yahoo Finance, and an LLM reasoning agent."
    ),
    version="0.1.0",
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


# ---------------------------------------------------------------------------
# Startup / Shutdown Events
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    """Log application startup."""
    logger.info("Financial AI Assistant is starting up …")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Log application shutdown."""
    logger.info("Financial AI Assistant is shutting down …")
