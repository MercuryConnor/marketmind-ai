# MarketMind AI — Repository Overview

## Top-Level
- plan.md — Project plan, architecture, run/deploy commands, and TODOs.
- README.md — Project overview and quickstart.
- requirements.txt — Python backend and tooling dependencies.
- tests — Test suite for backend components and pipelines.

## Backend (app/)
- app — Backend package root.
- main.py — FastAPI application creation, router registration, CORS, and lifespan hooks (entrypoint for the HTTP service).
- streamlit_app.py — Optional Streamlit demo UI for local inspection and demos.
- __init__.py — Package initializer.

### Agents
- agents — Agent implementations and orchestrators.
- financial_agent.py — Main orchestrator: detects tickers, selects tools (market data, RAG), executes tools, aggregates context, and runs LLM reasoning to produce insights.
- __init__.py — Package initializer.

### API
- routes.py — HTTP endpoints (`/health`, `/ask`), request validation, and wiring requests into the `FinancialAgent`.

### MCP (Model Context Protocol)
- mcp — MCP integration.
- mcp_server.py — Exposes tools via MCP so external LLM clients can call `get_stock_data` and `query_docs`; handles tool routing and access control.
- __init__.py — Package initializer.

### RAG (Retrieval-Augmented Generation)
- rag — Index build and query components.
- index_builder.py — Loads documents from financial_docs, chunks and embeds them (HuggingFace), builds and persists the FAISS vector index.
- query_engine.py — Loads persisted FAISS index, embeds the user query, retrieves top-K similar snippets, and returns snippet text with similarity scores.
- __init__.py — Package initializer.

### Services & Tools
- services — ETL and auxiliary services.
- data_pipeline.py — Data preparation and transformations used before indexing or ingestion.
- __init__.py — Package initializer.
- tools — Tool implementations callable by agents.
- market_data_tool.py — Fetches live market snapshots (Finnhub / yfinance), normalizes fields: `price`, `change_pct`, `market_cap`, `pe_ratio`, `timestamp`.
- __init__.py — Package initializer.

## Data
- data — Data and persisted artifacts.
- financial_docs — Source knowledge files (e.g., `pe_ratio_basics.txt`, `valuation_metrics.txt`) used to build the RAG index.
- faiss_index — Persisted vector store artifacts and metadata (`default__vector_store.json`, `docstore.json`, `index_store.json`, graph files, and FAISS binaries).

## Docs
- api.md — API contract, payload examples, and endpoint details.
- architecture.md — System design diagrams and component interactions.
- setup.md — Installation, environment variables, cache paths, and index build instructions.

## Frontend (frontend/)
- frontend — Next.js watchlist UI project (App Router + TypeScript).
- package.json — Frontend dependencies and npm scripts.
- package-lock.json — Lockfile for deterministic dependency install.
- layout.tsx — Root layout with fonts and providers.
- page.tsx — Main watchlist page.
- globals.css — Global CSS and design tokens.
- components — UI components:
  - theme-provider.tsx — Theme management.
  - sidebar.tsx — Sidebar navigation and ticker lists.
  - top-header.tsx — Top header with title, theme toggle, notifications.
  - search-bar.tsx — Search and add ticker UI.
  - watchlist-table.tsx — Watchlist table presentation.
  - quick-view-card.tsx — Stock detail card (price, change, chart).
  - sentiment-card.tsx — AI sentiment insight card.
  - ui — UI primitives (button, input, card, dropdown).
- watchlist.ts — Mock/watchlist sample data used for UI.
- utils.ts — Utility functions and API client helpers.
- next.config.mjs, tailwind.config.ts, tsconfig.json — Build and styling configuration.

## Tests
- test_api_routes.py — Tests for `/health` and `/ask` endpoints.
- test_financial_agent.py — Tests for agent tool selection, aggregation, and reasoning fallbacks.
- test_rag_pipeline.py — Tests for index build and retrieval.
- test_data_pipeline.py — Tests for data transformation and pipeline logic.
- test_market_data_tool.py — Tests for market data fetching and normalization.
- test_mcp_server.py — Tests for MCP tool exposure and invocation.

---
