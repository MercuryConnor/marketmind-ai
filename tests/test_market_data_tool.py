"""Unit tests for Finnhub market data tool."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.tools.market_data_tool import get_stock_data


class TestMarketDataTool(unittest.TestCase):
    """Validate Finnhub market tool behavior and error handling."""

    def test_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvironmentError):
                get_stock_data("AAPL")

    def test_get_stock_data_success(self) -> None:
        responses = [
            {"c": 195.5, "v": 12345678},
            {"marketCapitalization": 3000000},
            {"s": "ok", "c": [190.0, 195.5]},
        ]

        with patch.dict(os.environ, {"FINNHUB_API_KEY": "test-key"}, clear=True):
            with patch("app.tools.market_data_tool._request_json", side_effect=responses):
                result = get_stock_data("aapl")

        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["price"], 195.5)
        self.assertEqual(result["volume"], 12345678)
        self.assertEqual(result["market_cap"], 3000000000000.0)
        self.assertEqual(result["weekly_change"], 2.89)


if __name__ == "__main__":
    unittest.main()
