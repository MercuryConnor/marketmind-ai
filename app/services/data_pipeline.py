"""Financial AI Assistant - Data Cleaning Pipeline.

This module transforms raw Yahoo Finance-like payloads into a stable,
normalized structure for downstream AI processing.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def clean_stock_data(raw_data: Mapping[str, Any]) -> dict[str, Any]:
    """Clean and validate raw stock data payload.

    Args:
        raw_data: Raw dictionary-like data from Yahoo Finance or from
            the market data tool.

    Returns:
        A cleaned dictionary with stable key names and normalized value types.

    Raises:
        ValueError: If input is empty or missing essential symbol information.
        TypeError: If input is not mapping-like.
    """
    if not isinstance(raw_data, Mapping):
        logger.error("raw_data has invalid type: %s", type(raw_data).__name__)
        raise TypeError("raw_data must be a mapping/dictionary")

    if not raw_data:
        logger.error("raw_data is empty")
        raise ValueError("raw_data cannot be empty")

    symbol = _extract_symbol(raw_data)
    if symbol is None:
        logger.error("Missing symbol in raw_data: keys=%s", list(raw_data.keys()))
        raise ValueError("Missing required stock symbol")

    cleaned: dict[str, Any] = {
        "symbol": symbol,
        "price": _to_float(raw_data.get("price", raw_data.get("regularMarketPrice", raw_data.get("currentPrice")))),
        "market_cap": _to_float(raw_data.get("market_cap", raw_data.get("marketCap"))),
        "volume": _to_int(raw_data.get("volume", raw_data.get("regularMarketVolume"))),
        "weekly_change": _to_float(raw_data.get("weekly_change")),
    }

    logger.info("Cleaned stock data for %s", symbol)
    return cleaned


def calculate_metrics(cleaned_data: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate derived metrics from cleaned stock data.

    Args:
        cleaned_data: Output of ``clean_stock_data``.

    Returns:
        A dictionary containing derived metrics for reasoning.

    Raises:
        TypeError: If ``cleaned_data`` is not mapping-like.
        ValueError: If symbol is missing.
    """
    if not isinstance(cleaned_data, Mapping):
        logger.error("cleaned_data has invalid type: %s", type(cleaned_data).__name__)
        raise TypeError("cleaned_data must be a mapping/dictionary")

    symbol = cleaned_data.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        logger.error("cleaned_data is missing a valid symbol")
        raise ValueError("cleaned_data must include a valid symbol")

    weekly_change = _to_float(cleaned_data.get("weekly_change"))
    market_cap = _to_float(cleaned_data.get("market_cap"))
    volume = _to_int(cleaned_data.get("volume"))

    metrics: dict[str, Any] = {
        "weekly_trend": _weekly_trend(weekly_change),
        "market_cap_billions": round(market_cap / 1_000_000_000, 2) if market_cap is not None else None,
        "volume_millions": round(volume / 1_000_000, 2) if volume is not None else None,
        "data_quality_score": _data_quality_score(cleaned_data),
    }

    logger.info("Calculated derived metrics for %s", symbol)
    return metrics


def normalize_output(cleaned_data: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize cleaned data and derived metrics into final shape.

    Args:
        cleaned_data: Output of ``clean_stock_data``.
        metrics: Output of ``calculate_metrics``.

    Returns:
        Final normalized dictionary ready for API/agent consumption.

    Raises:
        TypeError: If either input is not mapping-like.
    """
    if not isinstance(cleaned_data, Mapping) or not isinstance(metrics, Mapping):
        logger.error(
            "normalize_output received invalid inputs: cleaned=%s metrics=%s",
            type(cleaned_data).__name__,
            type(metrics).__name__,
        )
        raise TypeError("cleaned_data and metrics must be mapping/dictionary types")

    normalized: dict[str, Any] = {
        "symbol": cleaned_data.get("symbol"),
        "price": cleaned_data.get("price"),
        "market_cap": cleaned_data.get("market_cap"),
        "volume": cleaned_data.get("volume"),
        "weekly_change": cleaned_data.get("weekly_change"),
        "metrics": dict(metrics),
    }

    logger.info("Normalized stock output for %s", normalized.get("symbol"))
    return normalized


def _extract_symbol(raw_data: Mapping[str, Any]) -> str | None:
    """Extract and standardize symbol from supported key variants."""
    value = raw_data.get("symbol", raw_data.get("ticker"))
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    return None


def _to_float(value: Any) -> float | None:
    """Safely convert value to float, returning ``None`` on failure."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Unable to convert value=%r to float", value)
        return None


def _to_int(value: Any) -> int | None:
    """Safely convert value to int, returning ``None`` on failure."""
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Unable to convert value=%r to int", value)
        return None


def _weekly_trend(weekly_change: float | None) -> str:
    """Convert weekly change numeric value to a trend label."""
    if weekly_change is None:
        return "unknown"
    if weekly_change > 0:
        return "up"
    if weekly_change < 0:
        return "down"
    return "flat"


def _data_quality_score(cleaned_data: Mapping[str, Any]) -> float:
    """Score completeness of key numeric fields on a 0-1 scale."""
    numeric_keys = ("price", "market_cap", "volume", "weekly_change")
    present = sum(1 for key in numeric_keys if cleaned_data.get(key) is not None)
    return round(present / len(numeric_keys), 2)
