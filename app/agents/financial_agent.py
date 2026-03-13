"""Financial reasoning agent for tool selection and response generation."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Callable, TypedDict, cast

from app.mcp.mcp_server import MCPToolExecutor
from app.rag.query_engine import query_financial_docs
from app.tools.market_data_tool import get_stock_data

logger = logging.getLogger(__name__)

TickerExtractor = re.compile(r"\b[A-Z]{1,5}\b")
StockToolCallable = Callable[[str], dict[str, Any]]
RagToolCallable = Callable[[str], dict[str, Any]]
ReasoningCallable = Callable[[str, dict[str, Any]], str]


class AgentContext(TypedDict):
    """Typed container for aggregated tool context."""

    query: str
    market_data: dict[str, Any] | None
    rag_context: dict[str, Any] | None
    errors: list[str]


@dataclass(frozen=True)
class ToolSelection:
    """Represents selected tools for a user query."""

    use_market_data: bool
    use_rag: bool
    symbol: str | None


class FinancialAgent:
    """Reasoning agent that routes and combines financial tool outputs.

    The agent follows a simple production-safe workflow:
    1. Analyze query and select tools.
    2. Execute selected tools with failure isolation.
    3. Aggregate context.
    4. Produce final analysis, with timeout fallback handling.
    """

    def __init__(
        self,
        stock_tool: StockToolCallable = get_stock_data,
        rag_tool: RagToolCallable = query_financial_docs,
        mcp_executor: MCPToolExecutor | None = None,
        reasoning_fn: ReasoningCallable | None = None,
        model_timeout_seconds: float = 8.0,
    ) -> None:
        """Initialize the financial agent with injectable dependencies.

        Args:
            stock_tool: Callable for market data retrieval.
            rag_tool: Callable for document retrieval.
            mcp_executor: Optional MCP executor for tool calls.
            reasoning_fn: Optional LLM reasoning function.
            model_timeout_seconds: Timeout applied to reasoning function.
        """
        if mcp_executor is not None:
            self._stock_tool = mcp_executor.get_stock_data
            self._rag_tool = mcp_executor.query_financial_docs
        else:
            self._stock_tool = stock_tool
            self._rag_tool = rag_tool
        self._reasoning_fn = reasoning_fn
        self._model_timeout_seconds = model_timeout_seconds

    def handle_query(self, query: str) -> dict[str, Any]:
        """Process a user query into structured financial insight.

        Args:
            query: User natural-language query.

        Returns:
            Structured response with analysis, tool data, and final insight.

        Raises:
            ValueError: If query is empty.
        """
        if not query.strip():
            logger.error("Invalid query provided to FinancialAgent")
            raise ValueError("query must be a non-empty string")

        normalized_query = query.strip()
        selection = self.select_tools(normalized_query)
        context = self.aggregate_context(normalized_query, selection)
        analysis = self._generate_analysis(normalized_query, context)

        insight = self._build_insight(normalized_query, context, analysis)

        response: dict[str, Any] = {
            "analysis": analysis,
            "data": context,
            "insight": insight,
        }
        logger.info(
            "Query processed: tools market=%s rag=%s errors=%d",
            selection.use_market_data,
            selection.use_rag,
            len(context.get("errors", [])),
        )
        return response

    def select_tools(self, query: str) -> ToolSelection:
        """Select tools based on query intent and ticker detection."""
        lower_query = query.lower()

        stock_keywords = (
            "price",
            "stock",
            "market cap",
            "volume",
            "perform",
            "ticker",
            "share",
        )
        rag_keywords = (
            "what is",
            "explain",
            "definition",
            "ratio",
            "valuation",
            "metric",
        )

        symbol = self._extract_symbol(query)
        use_market_data = bool(symbol) or any(keyword in lower_query for keyword in stock_keywords)
        use_rag = any(keyword in lower_query for keyword in rag_keywords)

        if not use_market_data and not use_rag:
            use_rag = True

        logger.info(
            "Tool selection for query=%r -> market_data=%s rag=%s symbol=%s",
            query,
            use_market_data,
            use_rag,
            symbol,
        )
        return ToolSelection(use_market_data=use_market_data, use_rag=use_rag, symbol=symbol)

    def aggregate_context(self, query: str, selection: ToolSelection) -> AgentContext:
        """Execute selected tools and aggregate context with safe fallbacks."""
        context: AgentContext = {
            "query": query,
            "market_data": None,
            "rag_context": None,
            "errors": [],
        }

        if selection.use_market_data:
            symbol = selection.symbol
            if not symbol:
                error = "No stock ticker detected for market data request"
                logger.warning(error)
                context["errors"].append(error)
            else:
                try:
                    context["market_data"] = self._stock_tool(symbol)
                except Exception as exc:
                    logger.exception("Market data tool failed for symbol=%s", symbol)
                    context["errors"].append(f"market_data_tool_failed: {exc}")

        if selection.use_rag:
            try:
                context["rag_context"] = self._rag_tool(query)
            except Exception as exc:
                logger.exception("RAG tool failed for query=%r", query)
                context["errors"].append(f"rag_tool_failed: {exc}")

        return context

    def _generate_analysis(self, query: str, context: AgentContext) -> str:
        """Generate analysis text using optional reasoning function with timeout."""
        if self._reasoning_fn is None:
            return self._default_reasoning(query, context)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                model_context = cast(dict[str, Any], context)
                future = executor.submit(self._reasoning_fn, query, model_context)
                return str(future.result(timeout=self._model_timeout_seconds))
        except FuturesTimeoutError:
            logger.error("Reasoning model timed out after %.2f seconds", self._model_timeout_seconds)
            return "Reasoning timed out; returning best-effort analysis from available tools."
        except Exception as exc:
            logger.exception("Reasoning model failed")
            return f"Reasoning failed ({exc}); returning best-effort analysis from available tools."

    def _default_reasoning(self, query: str, context: AgentContext) -> str:
        """Deterministic fallback reasoning used when no model is configured."""
        parts: list[str] = [f"Query interpreted: {query}."]

        market_data = context["market_data"]
        if market_data is not None:
            symbol = str(market_data.get("symbol", "unknown symbol"))
            price = market_data.get("price")
            weekly_change = market_data.get("weekly_change")
            parts.append(
                f"Market data available for {symbol} (price={price}, weekly_change={weekly_change})."
            )

        rag_context = context["rag_context"]
        if rag_context is not None:
            snippets_raw = rag_context.get("snippets")
            snippets = cast(list[Any], snippets_raw) if isinstance(snippets_raw, list) else []
            if snippets:
                parts.append(f"Retrieved {len(snippets)} knowledge snippets from financial documents.")

        errors = context["errors"]
        if errors:
            parts.append("Some tools failed; response uses partial context.")

        return " ".join(parts)

    def _build_insight(self, query: str, context: AgentContext, analysis: str) -> str:
        """Build concise final insight from available context and analysis."""
        market_data = context["market_data"]
        if market_data is not None and market_data.get("symbol"):
            symbol = str(market_data.get("symbol"))
            trend = market_data.get("weekly_change")
            return f"For {symbol}, current signals are mixed; weekly change={trend}. {analysis}"

        rag_context = context["rag_context"]
        if rag_context is not None:
            snippets_raw = rag_context.get("snippets")
            snippets = cast(list[Any], snippets_raw) if isinstance(snippets_raw, list) else []
            if snippets:
                first = snippets[0]
                if isinstance(first, dict):
                    first_item = cast(dict[str, Any], first)
                    text = str(first_item.get("text", "")).strip()
                    if text:
                        return f"Knowledge insight: {text}"

        return "Limited context available; provide a ticker or a more specific financial question."

    @staticmethod
    def _extract_symbol(query: str) -> str | None:
        """Extract a likely ticker symbol from user query."""
        matches = TickerExtractor.findall(query.upper())
        if not matches:
            return None

        blocked = {
            "WHAT",
            "IS",
            "HOW",
            "DID",
            "THE",
            "THIS",
            "THAT",
            "WITH",
            "FOR",
            "RATIO",
            "P",
            "E",
        }
        for candidate in matches:
            if candidate not in blocked:
                return candidate
        return None
