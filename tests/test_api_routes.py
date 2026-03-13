"""Unit tests for API routes."""

from __future__ import annotations

import unittest
from unittest.mock import patch
from typing import Any

from fastapi.testclient import TestClient

from app.main import app


class TestAPIRoutes(unittest.TestCase):
    """Validate API health and assistant endpoint behavior."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "running"})

    def test_ask_endpoint_success(self) -> None:
        mock_response: dict[str, Any] = {
            "analysis": "Test analysis",
            "data": {"market_data": {"symbol": "AAPL"}},
            "insight": "Test insight",
        }

        with patch("app.api.routes.financial_agent.handle_query", return_value=mock_response):
            response = self.client.post("/ask", json={"query": "How did AAPL perform this week?"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["analysis"], "Test analysis")
        self.assertIn("data", body)
        self.assertEqual(body["insight"], "Test insight")

    def test_ask_endpoint_rejects_empty_query(self) -> None:
        response = self.client.post("/ask", json={"query": "   "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Query must be a non-empty string")

    def test_ask_endpoint_invalid_json(self) -> None:
        response = self.client.post(
            "/ask",
            content="not-json",
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(response.status_code, 422)

    def test_ask_endpoint_ai_error(self) -> None:
        with patch("app.api.routes.financial_agent.handle_query", side_effect=RuntimeError("model failure")):
            response = self.client.post("/ask", json={"query": "How did AAPL perform this week?"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "AI processing error")


if __name__ == "__main__":
    unittest.main()
