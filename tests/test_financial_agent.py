"""Unit tests for the FinancialAgent orchestration workflow."""

from __future__ import annotations

import time
import unittest
from typing import Any

from app.agents.financial_agent import FinancialAgent
from app.mcp.mcp_server import MCPToolExecutor, create_mcp_server


def _stock_ok(symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "price": 200.0,
        "market_cap": 1200000000000.0,
        "volume": 10000000,
        "weekly_change": 2.5,
    }


def _rag_ok(query: str) -> dict[str, Any]:
    return {
        "query": query,
        "snippets": [
            {"text": "P/E ratio compares share price to earnings per share.", "score": 0.1}
        ],
    }


class TestFinancialAgent(unittest.TestCase):
    """Validate tool selection, context aggregation, and fallback behavior."""

    def test_selects_market_data_for_stock_query(self) -> None:
        agent = FinancialAgent(stock_tool=_stock_ok, rag_tool=_rag_ok)

        result = agent.handle_query("How did AAPL perform this week?")

        self.assertIsNotNone(result["data"]["market_data"])
        self.assertEqual(result["data"]["market_data"]["symbol"], "AAPL")

    def test_selects_rag_for_definition_query(self) -> None:
        agent = FinancialAgent(stock_tool=_stock_ok, rag_tool=_rag_ok)

        result = agent.handle_query("What is P/E ratio?")

        self.assertIsNotNone(result["data"]["rag_context"])
        self.assertGreater(len(result["data"]["rag_context"]["snippets"]), 0)

    def test_handles_market_data_tool_failure(self) -> None:
        def stock_fail(_: str) -> dict[str, Any]:
            raise RuntimeError("stock down")

        agent = FinancialAgent(stock_tool=stock_fail, rag_tool=_rag_ok)
        result = agent.handle_query("AAPL price")

        self.assertIsNone(result["data"]["market_data"])
        self.assertTrue(any("market_data_tool_failed" in err for err in result["data"]["errors"]))

    def test_handles_rag_tool_failure(self) -> None:
        def rag_fail(_: str) -> dict[str, Any]:
            raise RuntimeError("rag down")

        agent = FinancialAgent(stock_tool=_stock_ok, rag_tool=rag_fail)
        result = agent.handle_query("Explain valuation metrics")

        self.assertIsNone(result["data"]["rag_context"])
        self.assertTrue(any("rag_tool_failed" in err for err in result["data"]["errors"]))

    def test_reasoning_timeout_fallback(self) -> None:
        def slow_reasoning(_: str, __: dict[str, Any]) -> str:
            time.sleep(0.2)
            return "Should not return"

        agent = FinancialAgent(
            stock_tool=_stock_ok,
            rag_tool=_rag_ok,
            reasoning_fn=slow_reasoning,
            model_timeout_seconds=0.05,
        )

        result = agent.handle_query("AAPL price")

        self.assertIn("timed out", result["analysis"])

    def test_rejects_empty_query(self) -> None:
        agent = FinancialAgent(stock_tool=_stock_ok, rag_tool=_rag_ok)

        with self.assertRaises(ValueError):
            agent.handle_query("   ")

    def test_agent_calls_mcp_tools_successfully(self) -> None:
        calls = {"stock": 0, "rag": 0}

        def stock_mcp(symbol: str) -> dict[str, Any]:
            calls["stock"] += 1
            return _stock_ok(symbol)

        def rag_mcp(query_text: str) -> dict[str, Any]:
            calls["rag"] += 1
            return _rag_ok(query_text)

        executor = MCPToolExecutor(server=create_mcp_server(stock_tool=stock_mcp, rag_tool=rag_mcp))
        agent = FinancialAgent(mcp_executor=executor)

        result = agent.handle_query("What is P/E ratio for AAPL?")

        self.assertEqual(calls["stock"], 1)
        self.assertEqual(calls["rag"], 1)
        self.assertIsNotNone(result["data"]["market_data"])
        self.assertIsNotNone(result["data"]["rag_context"])


if __name__ == "__main__":
    unittest.main()
