"""
Financial AI Assistant — Market Data Tool.

Fetches real-time and historical stock data from Finnhub
using HTTP API calls and returns structured JSON-ready dictionaries.

Requires a ``FINNHUB_API_KEY`` environment variable.

Includes retry logic with exponential backoff to handle HTTP failures
and API rate limiting (HTTP 429).
"""

import logging
import os
import time
from typing import Any, cast

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

# API configuration
_FINNHUB_BASE_URL: str = "https://finnhub.io/api/v1"

# Retry configuration for transient failures and rate limits
_MAX_RETRIES: int = 3
_RETRY_BASE_DELAY: float = 2.0  # seconds
_HTTP_TIMEOUT_SECONDS: float = 10.0


def get_stock_data(symbol: str) -> dict[str, Any]:
    """Retrieve key market data for a given stock symbol.

    Fetches the current price, market capitalisation, trading volume,
    and approximate weekly price change from Finnhub.

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
        ConnectionError: If the Finnhub API is unreachable or
            rate limiting persists after retries.
        EnvironmentError: If FINNHUB_API_KEY is missing.
        RuntimeError: For any other unexpected failures.
    """
    if not symbol or not symbol.strip():
        logger.error("Empty stock symbol provided")
        raise ValueError("Stock symbol must be a non-empty string.")

    api_key = _get_finnhub_api_key()
    symbol = symbol.strip().upper()
    logger.info("Fetching stock data for %s", symbol)

    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return _fetch_stock_data(symbol, api_key)
        except ConnectionError as exc:
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
        except EnvironmentError:
            raise
        except Exception as exc:
            logger.error("Unexpected error fetching data for %s: %s", symbol, exc)
            raise RuntimeError(
                f"Failed to fetch stock data for {symbol}: {exc}"
            ) from exc

    raise ConnectionError(
        f"Finnhub API unavailable after {_MAX_RETRIES} retries for {symbol}"
    ) from last_exc


def _fetch_stock_data(symbol: str, api_key: str) -> dict[str, Any]:
    """Fetch stock data from Finnhub endpoints in a single attempt.

    Args:
        symbol: Uppercased, validated stock ticker.
        api_key: Finnhub API key.

    Returns:
        Structured stock data dictionary.

    Raises:
        ValueError: If no data is found for the symbol.
        ConnectionError: If the API call fails or is rate-limited.
    """
    with httpx.Client(timeout=_HTTP_TIMEOUT_SECONDS) as client:
        quote = _request_json(
            client,
            "/quote",
            {"symbol": symbol, "token": api_key},
        )
        profile = _request_json(
            client,
            "/stock/profile2",
            {"symbol": symbol, "token": api_key},
        )
        candles = _request_json(
            client,
            "/stock/candle",
            _build_candles_params(symbol, api_key),
        )

    price = _safe_float(quote, "c")
    volume = _safe_int(quote, "v")
    market_cap = _market_cap_from_profile(profile)
    weekly_change = _weekly_change_from_candles(candles)

    has_profile = bool(profile)
    has_candles = _candles_have_data(candles)
    if price is None and not has_profile and not has_candles:
        logger.warning("No data returned for symbol %s", symbol)
        raise ValueError(f"Invalid stock symbol or no data available: {symbol}")

    result: dict[str, Any] = {
        "symbol": symbol,
        "price": price,
        "market_cap": market_cap,
        "volume": volume,
        "weekly_change": weekly_change,
    }

    logger.info("Successfully fetched data for %s: price=%s", symbol, price)
    return result


def _get_finnhub_api_key() -> str:
    """Retrieve Finnhub API key from environment."""
    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        logger.error("Missing FINNHUB_API_KEY environment variable")
        raise EnvironmentError("FINNHUB_API_KEY is required")
    return api_key


def _request_json(client: httpx.Client, path: str, params: dict[str, Any]) -> dict[str, Any]:
    """Perform a GET request and return parsed JSON object."""
    url = f"{_FINNHUB_BASE_URL}{path}"
    try:
        response = client.get(url, params=params)
    except httpx.HTTPError as exc:
        logger.warning("HTTP request failed for %s: %s", path, exc)
        raise ConnectionError(f"Finnhub request failed for {path}: {exc}") from exc

    if response.status_code == 429:
        logger.warning("Finnhub rate limit hit for endpoint %s", path)
        raise ConnectionError("Finnhub rate limit exceeded")

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("Finnhub endpoint %s returned status %s", path, response.status_code)
        raise ConnectionError(f"Finnhub returned status {response.status_code} for {path}") from exc

    payload = response.json()
    if not isinstance(payload, dict):
        raise ConnectionError(f"Unexpected response format from Finnhub endpoint {path}")

    return cast(dict[str, Any], payload)


def _build_candles_params(symbol: str, api_key: str) -> dict[str, Any]:
    """Build query params for candle data covering recent trading days."""
    now = int(time.time())
    from_ts = now - (14 * 24 * 60 * 60)
    return {
        "symbol": symbol,
        "resolution": "D",
        "from": from_ts,
        "to": now,
        "token": api_key,
    }


def _market_cap_from_profile(profile: dict[str, Any]) -> float | None:
    """Extract market cap from Finnhub profile payload."""
    # Finnhub returns marketCapitalization typically in millions.
    market_cap_millions = _safe_float(profile, "marketCapitalization")
    if market_cap_millions is None:
        return None
    return round(market_cap_millions * 1_000_000, 2)


def _candles_have_data(candles: dict[str, Any]) -> bool:
    """Return True if candle payload includes close-price entries."""
    closes_raw = candles.get("c")
    closes = cast(list[Any], closes_raw) if isinstance(closes_raw, list) else []
    status = candles.get("s")
    return len(closes) >= 2 and status == "ok"


def _weekly_change_from_candles(candles: dict[str, Any]) -> float | None:
    """Calculate weekly percentage change from Finnhub candle payload."""
    closes_raw = candles.get("c")
    status = candles.get("s")
    closes = cast(list[Any], closes_raw) if isinstance(closes_raw, list) else []
    if status != "ok" or len(closes) < 2:
        return None

    try:
        normalized_closes = [float(value) for value in closes if value is not None]
    except (TypeError, ValueError):
        return None

    if len(normalized_closes) < 2:
        return None

    start_price = normalized_closes[0]
    end_price = normalized_closes[-1]
    if start_price == 0:
        return None

    change_pct = ((end_price - start_price) / start_price) * 100
    return round(change_pct, 2)


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


