"""
Financial AI Assistant — API Routes.

Defines all HTTP endpoints exposed by the application.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.financial_agent import FinancialAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter()
financial_agent = FinancialAgent()


class AskRequest(BaseModel):
    """Request body schema for the /ask endpoint."""

    query: str = Field(..., description="User financial query")


class AskResponse(BaseModel):
    """Response schema returned by the /ask endpoint."""

    analysis: str
    data: dict[str, Any]
    insight: str


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@router.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Return the current health status of the service.

    Returns:
        A JSON object with a ``status`` key.

    Example::

        GET /health
        → {"status": "running"}
    """
    logger.info("Health check requested")
    return {"status": "running"}


@router.post("/ask", response_model=AskResponse, tags=["assistant"])
async def ask_question(payload: AskRequest) -> AskResponse:
    """Handle financial assistant queries through the reasoning agent.

    Args:
        payload: Request payload containing the user query.

    Returns:
        Structured assistant response with analysis, data, and insight.

    Raises:
        HTTPException: For validation and processing failures.
    """
    query = payload.query.strip()
    if not query:
        logger.warning("Empty query received on /ask endpoint")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be a non-empty string",
        )

    try:
        result = financial_agent.handle_query(query)
    except ValueError as exc:
        logger.warning("Invalid query provided to financial agent: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Financial agent processing failed for /ask")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI processing error",
        ) from exc

    return AskResponse(**result)
