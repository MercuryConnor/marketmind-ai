"""Unit tests for app.services.data_pipeline."""

import unittest

from app.services.data_pipeline import (
    calculate_metrics,
    clean_stock_data,
    normalize_output,
)


class TestDataPipeline(unittest.TestCase):
    """Validate stock data cleaning and normalization behavior."""

    def test_clean_stock_data_with_standard_keys(self) -> None:
        raw = {
            "symbol": "aapl",
            "price": "190.2",
            "market_cap": "2900000000000",
            "volume": "54123000",
            "weekly_change": "-1.35",
        }

        cleaned = clean_stock_data(raw)

        self.assertEqual(cleaned["symbol"], "AAPL")
        self.assertEqual(cleaned["price"], 190.2)
        self.assertEqual(cleaned["market_cap"], 2900000000000.0)
        self.assertEqual(cleaned["volume"], 54123000)
        self.assertEqual(cleaned["weekly_change"], -1.35)

    def test_clean_stock_data_with_yahoo_keys(self) -> None:
        raw = {
            "symbol": "MSFT",
            "regularMarketPrice": 421.2,
            "marketCap": 3100000000000,
            "regularMarketVolume": 20000000,
        }

        cleaned = clean_stock_data(raw)

        self.assertEqual(cleaned["price"], 421.2)
        self.assertEqual(cleaned["market_cap"], 3100000000000.0)
        self.assertEqual(cleaned["volume"], 20000000)
        self.assertIsNone(cleaned["weekly_change"])

    def test_clean_stock_data_rejects_empty_payload(self) -> None:
        with self.assertRaises(ValueError):
            clean_stock_data({})

    def test_clean_stock_data_handles_corrupted_values(self) -> None:
        raw = {
            "symbol": "TSLA",
            "price": "not-a-number",
            "market_cap": None,
            "volume": "oops",
            "weekly_change": "3.2",
        }

        cleaned = clean_stock_data(raw)

        self.assertIsNone(cleaned["price"])
        self.assertIsNone(cleaned["market_cap"])
        self.assertIsNone(cleaned["volume"])
        self.assertEqual(cleaned["weekly_change"], 3.2)

    def test_calculate_metrics(self) -> None:
        cleaned = {
            "symbol": "AAPL",
            "price": 190.2,
            "market_cap": 2900000000000.0,
            "volume": 54123000,
            "weekly_change": -1.35,
        }

        metrics = calculate_metrics(cleaned)

        self.assertEqual(metrics["weekly_trend"], "down")
        self.assertEqual(metrics["market_cap_billions"], 2900.0)
        self.assertEqual(metrics["volume_millions"], 54.12)
        self.assertEqual(metrics["data_quality_score"], 1.0)

    def test_normalize_output(self) -> None:
        cleaned = {
            "symbol": "AAPL",
            "price": 190.2,
            "market_cap": 2900000000000.0,
            "volume": 54123000,
            "weekly_change": -1.35,
        }
        metrics = {
            "weekly_trend": "down",
            "market_cap_billions": 2900.0,
            "volume_millions": 54.12,
            "data_quality_score": 1.0,
        }

        normalized = normalize_output(cleaned, metrics)

        self.assertIn("metrics", normalized)
        self.assertEqual(normalized["symbol"], "AAPL")
        self.assertEqual(normalized["metrics"]["weekly_trend"], "down")


if __name__ == "__main__":
    unittest.main()
