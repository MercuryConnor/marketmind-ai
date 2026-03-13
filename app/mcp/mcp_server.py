"""MCP server and execution helpers for financial tools."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
from collections.abc import Coroutine
from typing import Any, Callable, cast

from mcp.server.fastmcp import FastMCP

from app.rag.query_engine import query_financial_docs
from app.tools.market_data_tool import get_stock_data

logger = logging.getLogger(__name__)

StockToolCallable = Callable[[str], dict[str, Any]]
RagToolCallable = Callable[[str], dict[str, Any]]


def create_mcp_server(
    stock_tool: StockToolCallable = get_stock_data,
    rag_tool: RagToolCallable = query_financial_docs,
) -> FastMCP:
    """Create MCP server exposing financial tool endpoints.

    Args:
        stock_tool: Callable used by ``get_stock_data`` MCP tool.
        rag_tool: Callable used by ``query_financial_docs`` MCP tool.

    Returns:
        Configured ``FastMCP`` server instance.
    """
    server = FastMCP("financial-ai-assistant")

    def get_stock_data_tool(symbol: str) -> str:
        """MCP tool wrapper for market data retrieval."""
        try:
            result = stock_tool(symbol)
        except Exception as exc:
            logger.exception("MCP get_stock_data tool failed for symbol=%s", symbol)
            raise RuntimeError(f"Tool execution failure: {exc}") from exc
        return json.dumps(result)

    def query_financial_docs_tool(query_text: str) -> str:
        """MCP tool wrapper for financial document retrieval."""
        try:
            result = rag_tool(query_text)
        except Exception as exc:
            logger.exception("MCP query_financial_docs tool failed for query=%r", query_text)
            raise RuntimeError(f"Tool execution failure: {exc}") from exc
        return json.dumps(result)

    cast(Any, server).add_tool(
        get_stock_data_tool,
        name="get_stock_data",
        description="Fetch stock market data for a ticker symbol",
    )
    cast(Any, server).add_tool(
        query_financial_docs_tool,
        name="query_financial_docs",
        description="Retrieve relevant financial knowledge snippets for a query",
    )

    logger.info("MCP server initialized with tools: get_stock_data, query_financial_docs")
    return server


class MCPToolExecutor:
    """Execute financial tools through an MCP server interface."""

    def __init__(self, server: FastMCP | None = None) -> None:
        """Initialize executor with a provided or default MCP server."""
        self._server = server if server is not None else create_mcp_server()

    def list_tools(self) -> list[str]:
        """Return names of currently registered MCP tools."""
        try:
            tools_result = self._resolve_awaitable(cast(Any, self._server).list_tools())
            return [tool.name for tool in tools_result]
        except Exception as exc:
            logger.exception("Failed to list MCP tools")
            raise ConnectionError(f"MCP connection issue while listing tools: {exc}") from exc

    def get_stock_data(self, symbol: str) -> dict[str, Any]:
        """Execute ``get_stock_data`` MCP tool."""
        return self._call_tool("get_stock_data", {"symbol": symbol})

    def query_financial_docs(self, query_text: str) -> dict[str, Any]:
        """Execute ``query_financial_docs`` MCP tool."""
        return self._call_tool("query_financial_docs", {"query_text": query_text})

    def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute tool and decode its JSON payload output.

        Raises:
            ConnectionError: If MCP connection/transport operations fail.
            RuntimeError: If tool execution fails or payload cannot be decoded.
        """
        try:
            result = self._resolve_awaitable(cast(Any, self._server).call_tool(tool_name, arguments))
        except Exception as exc:
            message = str(exc).lower()
            logger.exception("MCP call failed for tool=%s args=%s", tool_name, arguments)
            if "connection" in message or "connect" in message or "transport" in message:
                raise ConnectionError(f"MCP connection issue for tool {tool_name}: {exc}") from exc
            raise RuntimeError(f"MCP tool execution failure for {tool_name}: {exc}") from exc

        if not result:
            logger.error("MCP tool %s returned empty response", tool_name)
            raise RuntimeError(f"MCP tool {tool_name} returned empty response")

        for content in result:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    logger.exception("Invalid JSON response from MCP tool %s", tool_name)
                    raise RuntimeError(f"Invalid JSON response from MCP tool {tool_name}") from exc
                if isinstance(payload, dict):
                    return cast(dict[str, Any], payload)
                logger.error("Unexpected non-dict payload from MCP tool %s", tool_name)
                raise RuntimeError(f"Unexpected payload type from MCP tool {tool_name}")

        logger.error("No text payload in MCP response for tool %s", tool_name)
        raise RuntimeError(f"No text payload in MCP response for tool {tool_name}")

    @staticmethod
    def _resolve_awaitable(value: Any) -> Any:
        """Resolve awaitable values for environments where MCP APIs are async."""
        if inspect.isawaitable(value):
            coroutine = cast(Coroutine[Any, Any, Any], value)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coroutine)

            result_holder: dict[str, Any] = {}
            error_holder: dict[str, BaseException] = {}

            def _runner() -> None:
                try:
                    result_holder["result"] = asyncio.run(coroutine)
                except BaseException as exc:  # noqa: BLE001
                    error_holder["error"] = exc

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()

            if "error" in error_holder:
                raise error_holder["error"]

            return result_holder.get("result")
        return value
