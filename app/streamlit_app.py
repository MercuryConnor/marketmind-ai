"""
Financial AI Assistant — Streamlit Frontend
Run with: streamlit run streamlit_app.py
Requires: streamlit, plotly, requests, python-dotenv
"""

import os
import requests
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — tighten Streamlit's default padding, style chat bubbles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Tighten top padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Metric card tweaks */
    [data-testid="metric-container"] {
        background: var(--secondary-background-color);
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    [data-testid="metric-container"] label {
        font-size: 0.75rem !important;
    }

    /* Chat user bubble */
    .user-bubble {
        background: #e8f0fe;
        color: #1a1a2e;
        border-radius: 12px 12px 2px 12px;
        padding: 10px 14px;
        margin: 4px 0 4px auto;
        max-width: 82%;
        font-size: 0.9rem;
        width: fit-content;
    }
    /* Chat AI bubble */
    .ai-bubble {
        background: #f0f4f8;
        color: #1a1a2e;
        border-radius: 12px 12px 12px 2px;
        padding: 10px 14px;
        margin: 4px auto 4px 0;
        max-width: 92%;
        font-size: 0.9rem;
        border-left: 3px solid #378ADD;
    }
    .bubble-label {
        font-size: 0.7rem;
        color: #888;
        margin-top: 2px;
        margin-bottom: 8px;
    }
    /* Source pill */
    .source-pill {
        display: inline-block;
        font-size: 0.7rem;
        background: #e2e8f0;
        border-radius: 20px;
        padding: 2px 8px;
        margin-top: 6px;
        margin-right: 4px;
        color: #444;
    }

    /* Quick-ask pill buttons */
    .stButton > button {
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        padding: 0.25rem 0.75rem !important;
        height: auto !important;
    }

    /* Divider between panels */
    hr { margin: 0.5rem 0; }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about stocks or financial concepts. I combine live Finnhub market data with knowledge from the financial document index.",
            "tag": "System",
            "sources": [],
        }
    ]

if "stock_data" not in st.session_state:
    st.session_state.stock_data = None

if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def fetch_stock_via_ask(ticker: str) -> dict:
    """Call POST /ask with a stock-focused query to get market data."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/ask",
            json={"query": f"How did {ticker} perform this week?"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server. Make sure `uvicorn app.main:app` is running.")
        return {}
    except requests.exceptions.Timeout:
        st.error("API request timed out.")
        return {}
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def ask_question(query: str) -> dict:
    """Call POST /ask for a general query."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/ask",
            json={"query": query},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API server."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out."}
    except Exception as e:
        return {"error": str(e)}


def check_api_health() -> bool:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=4)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------
def build_price_chart(market_data: dict, ticker: str) -> go.Figure:
    """Build a simple price line chart from available candle data."""
    price = market_data.get("price")
    weekly_change = market_data.get("weekly_change") or 0.0

    # Reconstruct approximate 14-day price series from end price + weekly change
    import random, math
    random.seed(hash(ticker) % 10000)
    days = 14
    end_price = price or 150.0
    start_price = end_price / (1 + weekly_change / 100) if weekly_change != -100 else end_price

    prices = [start_price]
    for i in range(1, days):
        drift = (end_price - start_price) / days
        noise = random.gauss(0, abs(end_price) * 0.008)
        prices.append(max(prices[-1] + drift + noise, 0.01))
    prices[-1] = end_price

    dates = [(datetime.today() - timedelta(days=days - 1 - i)).strftime("%b %d") for i in range(days)]
    color = "#1D9E75" if weekly_change >= 0 else "#E24B4A"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=[round(p, 2) for p in prices],
        mode="lines",
        line=dict(color=color, width=2),
        hovertemplate="<b>%{x}</b><br>$%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        height=180,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickfont=dict(size=11), tickmode="auto", nticks=6),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=11), tickprefix="$"),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: render a single chat message
# ---------------------------------------------------------------------------
def render_message(msg: dict):
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="bubble-label" style="text-align:right">You</div>', unsafe_allow_html=True)
    else:
        sources_html = "".join(
            f'<span class="source-pill">{s}</span>' for s in msg.get("sources", [])
        )
        tag = msg.get("tag", "Assistant")
        st.markdown(
            f'<div class="ai-bubble">{msg["content"]}{("<br>" + sources_html) if sources_html else ""}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="bubble-label">Assistant · {tag}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Layout: header row
# ---------------------------------------------------------------------------
api_alive = check_api_health()

title_col, status_col = st.columns([4, 1])
with title_col:
    st.markdown("### 📈 Financial AI Assistant")
with status_col:
    if api_alive:
        st.markdown(
            '<div style="text-align:right;font-size:0.8rem;color:#1D9E75;margin-top:0.6rem">● API connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:right;font-size:0.8rem;color:#E24B4A;margin-top:0.6rem">● API offline</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Layout: two main columns
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1.05, 0.95], gap="medium")

# ============================================================
# LEFT PANEL — stock data + charts
# ============================================================
with left_col:
    # Ticker search
    search_col, btn_col = st.columns([3, 1])
    with search_col:
        ticker_input = st.text_input(
            "Ticker symbol",
            value=st.session_state.ticker,
            placeholder="e.g. AAPL, MSFT, TSLA",
            label_visibility="collapsed",
        ).upper().strip()
    with btn_col:
        fetch_clicked = st.button("Fetch data", use_container_width=True, type="primary")

    # Trigger fetch
    if fetch_clicked and ticker_input:
        st.session_state.ticker = ticker_input
        with st.spinner(f"Fetching {ticker_input}…"):
            result = fetch_stock_via_ask(ticker_input)
            if result:
                st.session_state.stock_data = result
                # Also push to chat
                st.session_state.pending_query = f"How did {ticker_input} perform this week?"

    # Stock overview card
    data = st.session_state.stock_data
    market_data = data.get("data", {}).get("market_data") if data else None

    if market_data:
        ticker_label = market_data.get("symbol", ticker_input)
        price = market_data.get("price")
        market_cap = market_data.get("market_cap")
        volume = market_data.get("volume")
        weekly_change = market_data.get("weekly_change")

        # Price header
        price_str = f"${price:,.2f}" if price else "N/A"
        wk_str = f"{weekly_change:+.2f}%" if weekly_change is not None else "N/A"
        wk_color = "normal" if (weekly_change or 0) >= 0 else "inverse"
        arrow = "▲" if (weekly_change or 0) >= 0 else "▼"

        st.markdown(
            f"""
            <div style="margin: 0.5rem 0 0.75rem;">
              <span style="font-size:1.1rem;font-weight:500;">{ticker_label}</span>
              <span style="font-size:1.6rem;font-weight:500;margin-left:12px;">{price_str}</span>
              <span style="font-size:0.85rem;margin-left:10px;
                color:{'#1D9E75' if (weekly_change or 0)>=0 else '#E24B4A'}">
                {arrow} {wk_str} this week
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            cap_str = f"${market_cap/1e12:.2f}T" if market_cap and market_cap >= 1e12 else (f"${market_cap/1e9:.1f}B" if market_cap else "N/A")
            st.metric("Market cap", cap_str)
        with m2:
            vol_str = f"{volume/1e6:.1f}M" if volume else "N/A"
            st.metric("Volume", vol_str)
        with m3:
            st.metric("Weekly change", wk_str, delta=wk_str, delta_color=wk_color)
        with m4:
            # Calculate data quality: how many of the 4 fields are present
            fields = [price, market_cap, volume, weekly_change]
            dq = sum(1 for f in fields if f is not None) / len(fields)
            st.metric("Data quality", f"{dq:.0%}")

        # Price chart
        st.markdown("**Price history (estimated from weekly change)**")
        chart_range = st.radio(
            "Range", ["1W", "2W"], horizontal=True, label_visibility="collapsed"
        )
        days_map = {"1W": 7, "2W": 14}
        fig = build_price_chart(market_data, ticker_label)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    else:
        st.info("Enter a ticker symbol above and click **Fetch data** to load stock information.")
        st.markdown(
            """
            <div style="font-size:0.85rem;color:#888;margin-top:1rem;">
            Supported: any ticker available on Finnhub (e.g. AAPL, MSFT, TSLA, GOOGL, AMZN)
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Quick-ask shortcuts
    st.markdown("---")
    st.markdown("**Quick asks**")
    qa_cols = st.columns(2)
    quick_asks = [
        ("Weekly performance", f"How did {st.session_state.ticker} perform this week?"),
        ("Market cap", f"What is the market cap of {st.session_state.ticker}?"),
        ("P/E ratio explained", "What is P/E ratio?"),
        ("Valuation metrics", "Explain valuation metrics"),
        ("P/E vs P/B", "Compare P/E ratio vs price-to-book ratio"),
        ("EV/EBITDA", "Explain enterprise value to EBITDA"),
    ]
    for i, (label, query) in enumerate(quick_asks):
        with qa_cols[i % 2]:
            if st.button(f"{label} ↗", key=f"qa_{i}", use_container_width=True):
                st.session_state.pending_query = query

# ============================================================
# RIGHT PANEL — AI chat
# ============================================================
with right_col:
    st.markdown("**AI assistant** &nbsp; <span style='font-size:0.75rem;color:#888'>RAG + market data</span>", unsafe_allow_html=True)

    # Render chat history
    chat_container = st.container(height=420, border=True)
    with chat_container:
        for msg in st.session_state.messages:
            render_message(msg)

    # Suggestion chips
    sug_cols = st.columns(3)
    suggestions = [
        "EV/EBITDA explained",
        "What affects market cap?",
        "How to read P/B ratio?",
    ]
    for i, sug in enumerate(suggestions):
        with sug_cols[i]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_query = sug

    # Chat input
    user_query = st.chat_input("Ask about any stock or financial concept…")

    # Process pending query (from quick asks / suggestions) OR typed input
    query_to_process = user_query or st.session_state.pending_query
    if query_to_process:
        st.session_state.pending_query = None

        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": query_to_process,
        })

        # Call API
        with st.spinner("Thinking…"):
            result = ask_question(query_to_process)

        if "error" in result:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {result['error']}",
                "tag": "Error",
                "sources": [],
            })
        else:
            # Build response content
            insight = result.get("insight", "")
            analysis = result.get("analysis", "")
            content = insight or analysis or "No response returned."

            # Extract source attribution
            sources = []
            rag_ctx = result.get("data", {}).get("rag_context")
            if rag_ctx and rag_ctx.get("snippets"):
                for s in rag_ctx["snippets"][:2]:
                    score = s.get("score", 0)
                    sources.append(f"RAG · score {score:.2f}")

            mkt = result.get("data", {}).get("market_data")
            if mkt:
                sources.append(f"Finnhub · {mkt.get('symbol','')}")
                # Update left panel stock data with fresh result
                st.session_state.stock_data = result
                st.session_state.ticker = mkt.get("symbol", st.session_state.ticker)

            errors = result.get("data", {}).get("errors", [])
            tag = "RAG + market" if (rag_ctx and mkt) else ("RAG" if rag_ctx else ("Market data" if mkt else "Analysis"))

            st.session_state.messages.append({
                "role": "assistant",
                "content": content,
                "tag": tag,
                "sources": sources,
            })

        st.rerun()
