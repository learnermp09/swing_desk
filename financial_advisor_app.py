"""
DecodixAI — Swing Desk
An Indian-equity swing-trade analyst powered by an Agno multi-agent team
(Groq LLM + DuckDuckGo research + live yfinance data), wrapped in a
trading-terminal styled Streamlit front end.

Run with:
    streamlit run financial_advisor_app.py
"""

import os
import datetime as dt

import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

from agno.agent import Agent
from agno.team import Team
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv(override=True)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

MODEL_ID = "qwen/qwen3.6-27b"

st.set_page_config(
    page_title="DecodixAI | Swing Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling — trading-terminal aesthetic
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0B0F14;
    --panel: #12181F;
    --panel-alt: #161D26;
    --border: #232D38;
    --text: #E8EDF2;
    --muted: #7C8B99;
    --gain: #3ECF8E;
    --loss: #FF5C5C;
    --amber: #F0B429;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }

/* ---- Ticker tape ---- */
.ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 0;
    margin-bottom: 1.4rem;
}
.ticker {
    display: inline-block;
    white-space: nowrap;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    padding-left: 100%;
    animation: ticker-scroll 28s linear infinite;
}
@keyframes ticker-scroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-100%); }
}
.ticker span.up   { color: var(--gain); }
.ticker span.down { color: var(--loss); }
.ticker span.sep  { color: var(--muted); padding: 0 18px; }

/* ---- Headline ---- */
.brand-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--amber);
    font-size: 0.72rem;
    margin-bottom: 0.35rem;
}
.brand-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.2rem;
    line-height: 1.15;
    margin-bottom: 0.2rem;
    color: var(--text);
}
.brand-sub { color: var(--muted); font-size: 0.95rem; margin-bottom: 1.6rem; }

/* ---- Cards ---- */
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.0rem 1.2rem;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 600;
    margin-top: 0.15rem;
}
.metric-delta { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; margin-top: 0.1rem; }
.metric-delta.up { color: var(--gain); }
.metric-delta.down { color: var(--loss); }

/* ---- Buttons ---- */
.stButton > button {
    background: var(--amber);
    color: #0B0F14;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    padding: 0.55rem 1.4rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(240, 180, 41, 0.25);
}

/* ---- Inputs ---- */
.stTextInput > div > div > input, .stSelectbox > div > div {
    background: var(--panel-alt) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: #0E141B;
    border-right: 1px solid var(--border);
}

/* ---- Analysis output panel ---- */
.analysis-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    font-size: 0.96rem;
    line-height: 1.65;
}
.analysis-panel h1, .analysis-panel h2, .analysis-panel h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
}
.analysis-panel strong { color: var(--amber); }

.session-pill {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 3px 10px;
    margin: 2px 4px 2px 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Live market data helpers (direct yfinance — fast, independent of the LLM)
# ---------------------------------------------------------------------------
def to_nse(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not ticker or "." in ticker:
        return ticker
    return f"{ticker}.NS"


@st.cache_data(ttl=60)
def get_index_snapshot():
    indices = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
    out = []
    for label, sym in indices.items():
        try:
            fi = yf.Ticker(sym).fast_info
            price, prev = fi.get("last_price"), fi.get("previous_close")
            if price is None or prev is None:
                continue
            chg = price - prev
            pct = (chg / prev) * 100 if prev else 0.0
            out.append((label, price, chg, pct))
        except Exception:
            continue
    return out


@st.cache_data(ttl=60)
def quick_quote(ticker: str):
    try:
        fi = yf.Ticker(to_nse(ticker)).fast_info
        price, prev = fi.get("last_price"), fi.get("previous_close")
        if price is None or prev is None:
            return None
        chg = price - prev
        pct = (chg / prev) * 100 if prev else 0.0
        return price, chg, pct
    except Exception:
        return None


def render_ticker():
    snapshot = get_index_snapshot()
    now_ist = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
    parts = []
    for label, price, chg, pct in snapshot:
        cls = "up" if chg >= 0 else "down"
        arrow = "▲" if chg >= 0 else "▼"
        parts.append(f'<span class="{cls}">{label} {price:,.2f} {arrow} {pct:+.2f}%</span>')
    if not parts:
        parts.append("<span>Market data unavailable</span>")
    parts.append(f'<span>IST {now_ist.strftime("%H:%M:%S")}</span>')
    ticker_html = '<span class="sep">•</span>'.join(parts)
    st.markdown(f'<div class="ticker-wrap"><div class="ticker">{ticker_html}</div></div>', unsafe_allow_html=True)


def render_quote_card(label: str, ticker: str):
    quote = quick_quote(ticker)
    with st.container():
        if quote is None:
            st.markdown(
                f'<div class="card"><div class="metric-label">{label} · {to_nse(ticker)}</div>'
                f'<div class="metric-value">—</div>'
                f'<div class="metric-delta">no data</div></div>',
                unsafe_allow_html=True,
            )
            return
        price, chg, pct = quote
        cls = "up" if chg >= 0 else "down"
        arrow = "▲" if chg >= 0 else "▼"
        st.markdown(
            f'<div class="card"><div class="metric-label">{label} · {to_nse(ticker)}</div>'
            f'<div class="metric-value">₹{price:,.2f}</div>'
            f'<div class="metric-delta {cls}">{arrow} {chg:+.2f} ({pct:+.2f}%)</div></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Agent team (built once, cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_team():
    web_agent = Agent(
        name="Web Search Agent",
        model=Groq(id=MODEL_ID),
        tools=[DuckDuckGoTools()],
        instructions=["Please do research on the given question and provide valuable finding"],
    )
    finance_agent = Agent(
        name="Finance Agent",
        model=Groq(id=MODEL_ID),
        tools=[YFinanceTools()],
        instructions=["Please analyze the user's financial query."],
    )
    team_manager = Team(
        name="Financial Advisor",
        model=Groq(id=MODEL_ID),
        members=[web_agent, finance_agent],
        instructions=["Provide summary and advice by collecting inputs from web_agent and finance_agent"],
    )
    return team_manager


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown('<div class="brand-eyebrow">Configure</div>', unsafe_allow_html=True)
    stock_a = st.text_input("Stock A", value="TCS")
    stock_b = st.text_input("Stock B (optional)", value="INFY")
    trade_style = st.selectbox("Trade style", ["swing trade", "intraday", "long-term hold"])

    st.markdown("---")
    st.markdown('<div class="brand-eyebrow">Stack</div>', unsafe_allow_html=True)
    st.caption(f"Model — Groq `{MODEL_ID}`")
    st.caption("Research — DuckDuckGo")
    st.caption("Market data — yfinance (NSE)")

    st.markdown("---")
    st.markdown('<div class="brand-eyebrow">Session history</div>', unsafe_allow_html=True)
    if st.session_state.history:
        for q in reversed(st.session_state.history[-8:]):
            st.markdown(f'<span class="session-pill">{q}</span>', unsafe_allow_html=True)
    else:
        st.caption("No queries yet this session.")

# ---------------------------------------------------------------------------
# Header + live ticker
# ---------------------------------------------------------------------------
render_ticker()

st.markdown(
    """
    <div class="brand-eyebrow">DecodixAI // Swing Desk</div>
    <div class="brand-title">Indian Equity Swing-Trade Analyst</div>
    <div class="brand-sub">Two agents, one verdict — grounded in live NSE data and web research.</div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Quick quote cards (fast, direct — independent of the LLM call below)
# ---------------------------------------------------------------------------
qc1, qc2 = st.columns(2)
with qc1:
    render_quote_card("Stock A", stock_a)
with qc2:
    if stock_b.strip():
        render_quote_card("Stock B", stock_b)

st.write("")

# ---------------------------------------------------------------------------
# Query + run
# ---------------------------------------------------------------------------
default_query = (
    f"{stock_a.strip()} vs {stock_b.strip()} for {trade_style}"
    if stock_b.strip()
    else f"{stock_a.strip()} for {trade_style}"
)
query = st.text_input("Ask the desk", value=default_query)

run_col, _ = st.columns([1, 5])
with run_col:
    run = st.button("Analyze ▶")

if run and query.strip():
    team_manager = build_team()
    with st.spinner("web_agent and finance_agent are on it..."):
        try:
            response = team_manager.run(query)
            content = response.content
        except Exception as exc:  # noqa: BLE001
            content = f"**Something went wrong reaching the agent team:**\n\n`{exc}`"

    st.session_state.history.append(query.strip())
    st.markdown(f'<div class="analysis-panel">{content}</div>', unsafe_allow_html=True)

st.write("")
st.caption(
    "Data can lag exchange feeds by several minutes and agent output may be incomplete — "
    "verify before placing a trade. Not investment advice."
)
