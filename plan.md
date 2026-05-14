# MarketMind AI - Project Plan & Architecture

**Project**: Financial AI Assistant with watchlist UI, RAG, and market intelligence backend.
**Status**: Alpha (Frontend working, Backend operational, RAG index built)
**Last Updated**: May 14, 2026

---

## 1. FRONTEND (Next.js Watchlist UI)

### Tech Stack
- **Framework**: Next.js 14.2.35 (App Router, TypeScript)
- **Styling**: Tailwind CSS v4 with `@tailwindcss/postcss` plugin
- **Component Library**: shadcn/ui (Radix UI primitives + lucide-react icons)
- **State Management**: Zustand (for global state if needed)
- **Charting**: Recharts for price charts
- **Tables**: TanStack React Table v8
- **Theme**: next-themes (light/dark/system modes)
- **Animation**: Framer Motion
- **HTTP Client**: Axios

### Project Structure
```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout with Inter + JetBrains_Mono fonts
│   │   ├── page.tsx            # Main watchlist page
│   │   └── globals.css         # Theme tokens & Tailwind imports
│   ├── components/
│   │   ├── ui/                 # shadcn components (button, dropdown, etc)
│   │   ├── layout/
│   │   │   ├── sidebar.tsx     # Left nav with ticker/market sections
│   │   │   └── top-header.tsx  # Top bar: title, theme toggle, notifications, profile
│   │   ├── watchlist/
│   │   │   ├── search-bar.tsx  # Search + filter/add ticker buttons
│   │   │   └── watchlist-table.tsx  # Core Watchlist table (NVDA, AAPL, TSLA, MSFT)
│   │   ├── dashboard/
│   │   │   └── quick-view-card.tsx  # Stock detail card (price, change, chart)
│   │   ├── sentiment/
│   │   │   └── sentiment-card.tsx   # AI Sentiment Insight (support/resistance/outlook)
│   │   └── theme-toggle.tsx    # Theme dropdown (hydration-safe)
│   ├── data/
│   │   └── watchlist.ts        # Mock data for table & quick view
│   ├── styles/
│   │   └── globals.css
│   └── lib/
│       └── (api clients, utils)
├── next.config.mjs             # Next.js config (no custom rules needed)
├── tailwind.config.ts          # Theme tokens via CSS vars
├── postcss.config.js           # PostCSS with Tailwind v4 plugin
├── tsconfig.json               # Path alias: @/* → src/*
└── package.json

```

### Design Tokens (CSS Variables)
- **Colors**:
  - `--mm-bg`: Background
  - `--mm-surface`: Primary surface
  - `--mm-panel`: Card/panel background
  - `--mm-border`: Border color
  - `--mm-text`: Primary text
  - `--mm-muted`: Secondary text
  - `--mm-accent`: Primary action color
  - `--mm-accent-soft`: Light accent
  - `--mm-positive`: Green (bullish, gain)
  - `--mm-negative`: Red (bearish, loss)
  - `--mm-warning`: Yellow/orange (caution)
- **Fonts**:
  - `--font-inter`: Main sans-serif (responsive)
  - `--font-jetbrains-mono`: Monospace for data

### Key Components
1. **Sidebar**: Market nav, ticker lists, trading info
2. **TopHeader**: "Your Watchlist" title, theme toggle, notifications, profile
3. **SearchBar**: Search input + Filter/Add buttons
4. **WatchlistTable**: 4 rows (NVDA, AAPL, TSLA, MSFT) with Ticker/Company/Price/Change/Market Cap/P/E/Sentiment/Actions
5. **QuickViewCard**: Selected stock detail (NVDA: $924.79, +6.21%, chart, buttons)
6. **SentimentCard**: AI analysis (Support $880, Resistance $950, Outlook Strong Buy)

### Known Issues & Fixes
- **Hydration mismatch** (FIXED): ThemeToggle now renders disabled placeholder during SSR, then fully hydrated on client
- All components properly handle client-side state with useEffect
- Next.js config must be `.mjs` not `.ts`
- Font imports use `next/font` with Inter & JetBrains_Mono (Geist removed)

### Running
```bash
cd frontend
npm install
npm run dev  # Runs on http://localhost:3000
```

---

## 2. RAG (Retrieval-Augmented Generation) Architecture

### Overview
Retrieves relevant financial documents to augment agent reasoning. Powered by FAISS vector DB + HuggingFace embeddings.

### Embedding Model
- **Model**: BAAI/bge-small-en-v1.5
- **Dimensions**: 384
- **Source**: HuggingFace Hub
- **Cached at**: `$HF_HOME` or `TRANSFORMERS_CACHE`
- **Downloads on first index build** (~50MB)

### Index Building
**File**: `app/rag/index_builder.py`

```python
build_financial_index(
    documents_dir="data/financial_docs/",  # Source .txt & .md files
    persist_dir="data/faiss_index/",       # Output index
    embedding_dim=384,
    embed_model_name="BAAI/bge-small-en-v1.5"
)
```

**Process**:
1. Load documents from `data/financial_docs/` (pe_ratio_basics.txt, valuation_metrics.txt, etc.)
2. Split into chunks via LlamaIndex SimpleDirectoryReader
3. Embed each chunk with HuggingFace model
4. Build FAISS index (Flat search, no quantization)
5. Persist to disk as vector store + metadata

### Query Engine
**File**: `app/rag/query_engine.py`

```python
query_financial_docs(
    query_text="What is a P/E ratio?",
    top_k=3  # Return top 3 relevant snippets
) → {"query": "...", "snippets": [{"text": "...", "score": 0.85}, ...]}
```

**Process**:
1. Load persisted FAISS index
2. Embed user query with same model
3. Retrieve top-k similar vectors (cosine similarity)
4. Return text + similarity scores

### Document Sources
- `data/financial_docs/pe_ratio_basics.txt` — P/E ratio concepts
- `data/financial_docs/valuation_metrics.txt` — Valuation techniques
- (Expandable: add more .txt/.md files, reindex)

### Cache Management
- Set env vars to control HF cache:
  ```bash
  export LLAMA_INDEX_CACHE_DIR=/path/to/cache
  export TRANSFORMERS_CACHE=/path/to/hf_cache
  export SENTENCE_TRANSFORMERS_HOME=/path/to/sbert_cache
  ```
- Clear corrupted cache: `rm -rf ~/.cache/huggingface/`
- Rebuild index: `python -c "from app.rag.index_builder import build_financial_index; build_financial_index()"`

---

## 3. BACKEND (FastAPI Agent)

### Overview
HTTP service with reasoning agent that routes queries to tools, aggregates context, and generates insights.

### Tech Stack
- **Framework**: FastAPI 0.115.6 + Uvicorn 0.34.0
- **LLM Framework**: LlamaIndex 0.14.21
- **Data**: Pydantic 2.10.4 for validation
- **Vector DB**: FAISS (CPU only)
- **API Integration**: Finnhub (live market data), OpenAI (optional LLM)
- **MCP**: Model Context Protocol for tool exposure
- **Config**: python-dotenv for env vars

### Architecture

```
User Query (HTTP POST /ask)
  ↓
FastAPI Router → FinancialAgent
  ↓
  ├─→ Tool Selection (analyze query for ticker/context)
  │   ├─→ Stock Tool (Finnhub API)
  │   └─→ RAG Tool (FAISS query engine)
  ↓
  ├─→ Tool Execution (concurrent, isolated)
  │   ├─→ get_stock_data(ticker) → {price, change%, pe, etc}
  │   └─→ query_financial_docs(query) → {snippets}
  ↓
  ├─→ Context Aggregation
  │   └─→ AgentContext {query, market_data, rag_context, errors}
  ↓
  ├─→ LLM Reasoning (timeout: 8s)
  │   └─→ Generate analysis + insight from context
  ↓
  └─→ Response (JSON)
      {analysis: "...", data: {...}, insight: "..."}
```

### Core Components

#### 1. FinancialAgent (`app/agents/financial_agent.py`)
Main orchestrator:
- **Workflow**: query → select tools → execute → aggregate → reason → insight
- **Tool Selection Logic**:
  - Detect ticker symbols (regex: `\b[A-Z]{1,5}\b`)
  - Route to stock tool if ticker found
  - Route to RAG tool for concept/valuation queries
- **Error Isolation**: Tool failures logged but don't crash response
- **Timeout Handling**: Reasoning limited to 8s; fallback if exceeded
- **Dependencies Injection**: Accept custom stock_tool, rag_tool, mcp_executor, reasoning_fn

#### 2. Market Data Tool (`app/tools/market_data_tool.py`)
```python
get_stock_data(ticker: str) → {
    "ticker": "NVDA",
    "price": 924.79,
    "change_pct": 6.21,
    "market_cap": 2.24e12,
    "pe_ratio": 66.21,
    "timestamp": "2026-05-14T10:30:00Z"
}
```
- Uses Finnhub API (requires `FINNHUB_API_KEY` env var)
- Returns structured market snapshot

#### 3. RAG Query Tool (`app/rag/query_engine.py`)
- Wraps index.query() with error handling
- Returns relevant text snippets ranked by similarity
- Used as context for LLM reasoning

#### 4. MCP Server (`app/mcp/mcp_server.py`)
- Exposes tools via Model Context Protocol
- Allows external LLM clients to call stock_data & query_docs
- Wrapper: MCPToolExecutor handles tool routing

### API Endpoints

#### Health Check
```
GET /health
→ {"status": "running"}
```

#### Ask Question
```
POST /ask
Request:  {"query": "What is a P/E ratio for NVDA?"}
Response: {
  "analysis": "P/E ratio is price-to-earnings...",
  "data": {
    "market_data": {...},
    "rag_snippets": [...]
  },
  "insight": "NVDA P/E of 66.21 is high compared to..."
}
```

### Configuration & Startup

**Main App** (`app/main.py`):
- FastAPI + CORS middleware
- Lifespan hooks: check Finnhub key, validate FAISS index
- Router includes /health, /ask

**Running**:
```bash
export FINNHUB_API_KEY=your_key_here
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Dependencies (Python)
```
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
llama-index==0.14.21
llama-index-embeddings-huggingface==0.7.0
llama-index-vector-stores-faiss==0.6.0
faiss-cpu==1.9.0.post1
yfinance==0.2.54  (alternative market data)
openai==1.59.5     (optional: GPT-based reasoning)
mcp==1.2.0
python-dotenv==1.0.1
httpx==0.28.1
streamlit==1.46.1  (optional: UI)
plotly==6.0.1
requests==2.34.1
```

### Environment Variables
```
FINNHUB_API_KEY=sk_...         # Required for live market data
OPENAI_API_KEY=sk-...          # Optional: for LLM reasoning
LLAMA_INDEX_CACHE_DIR=/path    # HF embedding cache
TRANSFORMERS_CACHE=/path
SENTENCE_TRANSFORMERS_HOME=/path
```

---

## 4. DATA LAYER

### Documents
- **Location**: `data/financial_docs/`
- **Format**: Plain text (.txt) or Markdown (.md)
- **Examples**:
  - pe_ratio_basics.txt
  - valuation_metrics.txt
- **Expandable**: Add more docs, rebuild index

### FAISS Index
- **Location**: `data/faiss_index/`
- **Contents**: Vector embeddings, metadata, index.faiss (binary)
- **Generated by**: `app/rag/index_builder.py::build_financial_index()`
- **Size**: ~50-100MB depending on doc count
- **Refresh**: Delete `data/faiss_index/` and rebuild if outdated

---

## 5. TESTING

**Test Files**:
- `tests/test_api_routes.py` — /health, /ask endpoints
- `tests/test_financial_agent.py` — Agent tool selection, context aggregation
- `tests/test_rag_pipeline.py` — Index build, query, retrieval
- `tests/test_market_data_tool.py` — Stock data fetching
- `tests/test_data_pipeline.py` — Data transformations
- `tests/test_mcp_server.py` — MCP tool exposure

**Run**:
```bash
pytest tests/ -v
```

---

## 6. DEPLOYMENT NOTES

### Development
```bash
# Terminal 1: Backend
cd marketmind-ai
export FINNHUB_API_KEY=...
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd frontend
npm run dev  # http://localhost:3000

# Terminal 3: Optional Streamlit UI
streamlit run app/streamlit_app.py
```

### Production
- **Backend**: Deploy with uvicorn/gunicorn on cloud (AWS/GCP/Heroku)
- **Frontend**: Build & deploy to Vercel, Netlify, or CDN
- **FAISS Index**: Persist to shared storage (S3, mounted volume)
- **Env Vars**: Set via CI/CD secrets
- **CORS**: Adjust allowed origins in FastAPI config

### Known Issues & Fixes
1. **HF Cache Corruption**: Clear cache and rebuild index
2. **Missing Finnhub Key**: Service starts but /ask fails gracefully
3. **Empty FAISS Index**: Service warns on startup; add docs and rebuild
4. **Hydration Mismatch (Frontend)**: Fixed in ThemeToggle with client-side only rendering

---

## 7. NEXT STEPS

### Short Term
- [ ] Wire frontend to backend API (axios client for /ask endpoint)
- [ ] Implement real-time market data in watchlist table
- [ ] Add search/filter functionality with debounce
- [ ] Test table row interactions (click for detail modal)

### Medium Term
- [ ] Deploy backend to cloud (AWS Lambda, GCP Cloud Run, etc)
- [ ] Deploy frontend to Vercel/Netlify
- [ ] Add authentication (JWT, OAuth)
- [ ] Expand financial docs (add earnings, sector analysis, etc)
- [ ] Integrate real LLM (OpenAI, Anthropic) for reasoning

### Long Term
- [ ] Add portfolio management UI
- [ ] Real-time alerts based on sentiment
- [ ] Multi-asset class support (crypto, commodities, bonds)
- [ ] Advanced charting and technical analysis
- [ ] Mobile app (React Native)
- [ ] Backtesting framework for strategies

---

## 8. KEY CONTACTS & REFERENCES

**Author**: MercuryConnor
**Repo**: https://github.com/MercuryConnor/marketmind-ai
**Branch**: main

### Documentation
- `docs/api.md` — API contract details
- `docs/architecture.md` — System design
- `docs/setup.md` — Installation & configuration
- `README.md` — Project overview

---

## 9. QUICK COMMANDS REFERENCE

```bash
# Index
python -c "from app.rag.index_builder import build_financial_index; build_financial_index()"

# Backend
uvicorn app.main:app --reload

# Frontend
npm run dev

# Tests
pytest tests/ -v

# Format/Lint
black app/
mypy app/
eslint src/
```

---

**Last Verified**: May 14, 2026
**Status**: All systems operational ✅
