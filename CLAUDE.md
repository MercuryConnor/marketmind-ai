# MarketMind AI - Project Memory

## Commands

### Backend (Root)
- **Install Dependencies**: `pip install -r requirements.txt`
- **Build RAG Index**: `python -c "from app.rag.index_builder import build_financial_index; build_financial_index()"`
- **Run API**: `uvicorn app.main:app --reload`
- **Run Tests**: `pytest tests/ -v`
- **MCP Server**: `python app/mcp/mcp_server.py`

### Frontend (`/frontend`)
- **Install Dependencies**: `npm install`
- **Dev Server**: `npm run dev`
- **Build**: `npm run build`
- **Lint**: `npm run lint`

## Development Standards

### Backend (Python)
- **Style**: PEP 8 compliance.
- **Validation**: Use Pydantic models for all API requests and responses.
- **Docs**: Use Google-style docstrings for complex logic.
- **Error Handling**: Implement specific exceptions for tool failures (MarketData, RAG).

### Frontend (Next.js)
- **Framework**: Next.js 14.2+ (App Router).
- **Styling**: Tailwind CSS v4 + shadcn/ui.
- **Icons**: Lucide React.
- **Hydration**: Ensure all components are hydration-safe for theme switching.
- **CRITICAL**: Refer to `node_modules/next/dist/docs/` for version-specific breaking changes.

## File Structure
- `app/`: FastAPI backend and LlamaIndex agents.
- `frontend/`: Next.js application.
- `data/`: RAG documents and FAISS index.
- `tests/`: Python test suite.
- `docs/`: Architectural and setup documentation.
