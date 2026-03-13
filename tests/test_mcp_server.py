"""Unit tests for MCP server and tool executor."""

from __future__ import annotations

import unittest
from typing import Any

from app.mcp.mcp_server import MCPToolExecutor, create_mcp_server


def _stock_ok(symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "price": 100.0,
        "market_cap": 500000000000.0,
        "volume": 123456,
        "weekly_change": 1.5,
    }


def _rag_ok(query_text: str) -> dict[str, Any]:
    return {
        "query": query_text,
        "snippets": [{"text": "P/E ratio compares price to earnings.", "score": 0.1}],
    }


class TestMCPServer(unittest.TestCase):
    """Validate MCP tool registration and execution behavior."""

    def test_tools_are_registered(self) -> None:
        executor = MCPToolExecutor(server=create_mcp_server(stock_tool=_stock_ok, rag_tool=_rag_ok))

        tools = executor.list_tools()

        self.assertIn("get_stock_data", tools)
        self.assertIn("query_financial_docs", tools)

    def test_get_stock_data_via_mcp(self) -> None:
        executor = MCPToolExecutor(server=create_mcp_server(stock_tool=_stock_ok, rag_tool=_rag_ok))

        result = executor.get_stock_data("AAPL")

        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["price"], 100.0)

    def test_query_financial_docs_via_mcp(self) -> None:
        executor = MCPToolExecutor(server=create_mcp_server(stock_tool=_stock_ok, rag_tool=_rag_ok))

        result = executor.query_financial_docs("What is P/E ratio?")

        self.assertIn("snippets", result)
        self.assertGreater(len(result["snippets"]), 0)

    def test_tool_execution_failure_is_handled(self) -> None:
        def stock_fail(_: str) -> dict[str, Any]:
            raise RuntimeError("upstream unavailable")

        executor = MCPToolExecutor(server=create_mcp_server(stock_tool=stock_fail, rag_tool=_rag_ok))

        with self.assertRaises(RuntimeError):
            executor.get_stock_data("AAPL")


if __name__ == "__main__":
    unittest.main()
