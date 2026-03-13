"""
Financial AI Assistant — Market Data Tool.

Fetches real-time and historical stock data from Yahoo Finance
using the ``yfinance`` library and returns structured JSON-ready
dictionaries.

Includes retry logic with exponential backoff to handle Yahoo Finance
rate limiting (HTTP 429).
"""

import json
import logging
import time
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

# Retry configuration for rate-limit (429) responses
_MAX_RETRIES: int = 3
_RETRY_BASE_DELAY: float = 2.0  # seconds


def get_stock_data(symbol: str) -> dict[str, Any]:
    """Retrieve key market data for a given stock symbol.

    Fetches the current price, market capitalisation, trading volume,
    and approximate weekly price change from Yahoo Finance.

    Args:
        symbol: A valid stock ticker symbol (e.g. ``"AAPL"``).

    Returns:
        A dictionary with the following structure::

            {
                "symbol": "AAPL",
                "price": 190.2,
                "market_cap": 2950000000000,
                "volume": 54123000,
                "weekly_change": -1.35
            }

    Raises:
        ValueError: If the symbol is empty or no data is found for it.
        ConnectionError: If the Yahoo Finance API is unreachable or
            rate limiting persists after retries.
        RuntimeError: For any other unexpected failures.
    """
    # ---- Input validation ------------------------------------------------
    if not symbol or not symbol.strip():
        logger.error("Empty stock symbol provided")
        raise ValueError("Stock symbol must be a non-empty string.")

    symbol = symbol.strip().upper()
    logger.info("Fetching stock data for %s", symbol)

    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return _fetch_stock_data(symbol)
        except (json.JSONDecodeError, ConnectionError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Attempt %d/%d for %s failed (%s), retrying in %.1fs",
                    attempt, _MAX_RETRIES, symbol, type(exc).__name__, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "All %d attempts exhausted for %s", _MAX_RETRIES, symbol
                )
        except ValueError:
            raise
        except Exception as exc:
            logger.error("Unexpected error fetching data for %s: %s", symbol, exc)
            raise RuntimeError(
                f"Failed to fetch stock data for {symbol}: {exc}"
            ) from exc

    raise ConnectionError(
        f"Yahoo Finance API unavailable after {_MAX_RETRIES} retries for {symbol}"
    ) from last_exc


def _fetch_stock_data(symbol: str) -> dict[str, Any]:
    """Internal helper that performs a single fetch attempt.

    Args:
        symbol: Uppercased, validated stock ticker.

    Returns:
        Structured stock data dictionary.

    Raises:
        ValueError: If no data is found for the symbol.
        json.JSONDecodeError: If the API returns an unparseable response
            (typically a 429 rate-limit page).
    """
    ticker = yf.Ticker(symbol)
    info: dict[str, Any] = ticker.info  # may raise JSONDecodeError on 429

    # Yahoo Finance returns minimal info for invalid tickers
    if not info or (
        info.get("trailingPegRatio") is None
        and info.get("regularMarketPrice") is None
    ):
        hist = ticker.history(period="5d")
        if hist.empty:
            logger.warning("No data returned for symbol %s", symbol)
            raise ValueError(
                f"Invalid stock symbol or no data available: {symbol}"
            )

    # ---- Extract core metrics --------------------------------------------
    price = _safe_float(info, "regularMarketPrice", "currentPrice")
    market_cap = _safe_float(info, "marketCap")
    volume = _safe_int(info, "regularMarketVolume", "volume")
    weekly_change = _calculate_weekly_change(ticker)

    result: dict[str, Any] = {
        "symbol": symbol,
        "price": price,
        "market_cap": market_cap,
        "volume": volume,
        "weekly_change": weekly_change,
    }

    logger.info("Successfully fetched data for %s: price=%s", symbol, price)
    return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_float(info: dict[str, Any], *keys: str) -> float | None:
    """Return the first truthy float value found among *keys*, or None.

    Args:
        info: The ticker info dictionary.
        *keys: One or more dictionary keys to try in order.

    Returns:
        The value as a float, or ``None`` if none of the keys yield a
        valid number.
    """
    for key in keys:
        value = info.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _safe_int(info: dict[str, Any], *keys: str) -> int | None:
    """Return the first truthy int value found among *keys*, or None.

    Args:
        info: The ticker info dictionary.
        *keys: One or more dictionary keys to try in order.

    Returns:
        The value as an int, or ``None`` if none of the keys yield a
        valid number.
    """
    for key in keys:
        value = info.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _calculate_weekly_change(ticker: yf.Ticker) -> float | None:
    """Calculate the approximate percentage change over the last trading week.

    Args:
        ticker: A ``yfinance.Ticker`` instance.

    Returns:
        The weekly percentage change rounded to two decimal places,
        or ``None`` if insufficient data is available.
    """
    try:
        hist = ticker.history(period="5d")
        if hist.empty or len(hist) < 2:
            logger.warning(
                "Insufficient history to compute weekly change for %s",
                ticker.ticker,
            )
            return None

        start_price: float = float(hist["Close"].iloc[0])
        end_price: float = float(hist["Close"].iloc[-1])

        if start_price == 0:
            return None

        change_pct: float = ((end_price - start_price) / start_price) * 100
        return round(change_pct, 2)

    except Exception as exc:
        logger.warning("Could not compute weekly change: %s", exc)
        return None
