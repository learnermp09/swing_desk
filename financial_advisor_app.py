"""
DecodixAI — Swing Desk

An educational AI-powered US stock swing-trading learning assistant built with
an Agno multi-agent architecture, Groq LLM, DuckDuckGo research, and live
Yahoo Finance market data.

This application is designed exclusively for learning and experimentation with
AI-assisted market analysis. It is NOT intended to provide financial,
investment, or trading advice and should never be used as the sole basis for
real trading or investment decisions.

Run:
    streamlit run financial_advisor_app.py
"""

import os
import datetime as dt
import re

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
load_dotenv()


MODEL_ID = "qwen/qwen3.6-27b"

st.set_page_config(
    page_title="DecodixAI | US Swing Trading Learning Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.info(
    "📚 **Educational Use Only**\n\n"
    "This application is designed exclusively for learning AI-assisted "
    "US stock swing-trading analysis. It is **not** intended for live trading, "
    "investment recommendations, or financial advice. Always conduct your own "
    "research before making investment decisions.\n\n"
    "Developed **@DecodixAI** by *Mrityunjay Pathak*"
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #F5F5F5;
    --panel: #EBEBEB;
    --panel-alt: #E0E0E0;
    --border: #D0D0D0;
    --text: #2C2C2C;
    --muted: #6B6B6B;
    --gain: #2E7D32;
    --loss: #C62828;
    --amber: #E6A817;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); }
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }

/* ---- Header with logo ---- */
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.header-left {
    flex: 1;
}
.header-right {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.header-right img {
    max-height: 60px;
    width: auto;
}
.brand-eyebrow-header {
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--amber);
    font-size: 0.72rem;
    margin-bottom: 0.1rem;
}
.brand-title-header {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.8rem;
    line-height: 1.15;
    color: var(--text);
}
.brand-sub-header {
    color: var(--muted);
    font-size: 0.85rem;
}

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
    color: var(--text);
}
.metric-delta { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; margin-top: 0.1rem; }
.metric-delta.up { color: var(--gain); }
.metric-delta.down { color: var(--loss); }

/* ---- Buttons ---- */
.stButton > button {
    background: var(--amber);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    padding: 0.55rem 1.4rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(230, 168, 23, 0.3);
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
    background: #E8E8E8;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown, 
section[data-testid="stSidebar"] .stCaption {
    color: var(--text);
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
    color: var(--text);
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
    color: var(--text);
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 3px 10px;
    margin: 2px 4px 2px 0;
}

/* Info and warning boxes */
.stInfo, .stWarning {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
.stInfo [data-testid="stMarkdownContainer"], 
.stWarning [data-testid="stMarkdownContainer"] {
    color: var(--text) !important;
}

/* Sidebar text color fixes */
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label {
    color: var(--text) !important;
}

/* Spinner color */
.stSpinner > div {
    border-color: var(--amber) !important;
}

/* Main content area */
.main > div {
    background: var(--bg);
}

/* Container backgrounds */
.stContainer {
    background: var(--bg);
}

/* Column backgrounds */
.stColumn {
    background: var(--bg);
}

/* Selectbox dropdown */
.stSelectbox > div > div {
    background: var(--panel-alt) !important;
}

/* Make sure all text is readable */
.stMarkdown, .stCaption, .stText, .stTextInput label {
    color: var(--text) !important;
}

/* Info box styling */
.stAlert {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* Tabs if any */
.stTabs [data-baseweb="tab-list"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
}

/* ---- Patience message ---- */
.patience-message {
    font-size: 0.85rem;
    color: var(--muted);
    font-style: italic;
    margin-top: 0.3rem;
    text-align: center;
}

/* ---- Rate limit error message ---- */
.rate-limit-error {
    background: #FFF3E0;
    border: 1px solid #FFB74D;
    border-left: 4px solid #FF6F00;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    color: #2C2C2C;
}
.rate-limit-error strong {
    color: #E65100;
}
.rate-limit-error .timer {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #BF360C;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Live market data helpers (direct yfinance — fast, independent of the LLM)
# ---------------------------------------------------------------------------
def clean_ticker(ticker: str) -> str:
    return ticker.strip().upper()


@st.cache_data(ttl=60)
def get_index_snapshot():
    indices = {"S&P 500": "^GSPC", "DOW JONES": "^DJI", "NASDAQ": "^IXIC"}
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
        fi = yf.Ticker(clean_ticker(ticker)).fast_info
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
    now_et = dt.datetime.utcnow() - dt.timedelta(hours=4)  # approx. US Eastern (ET, no DST handling)
    parts = []
    for label, price, chg, pct in snapshot:
        cls = "up" if chg >= 0 else "down"
        arrow = "▲" if chg >= 0 else "▼"
        parts.append(f'<span class="{cls}">{label} {price:,.2f} {arrow} {pct:+.2f}%</span>')
    if not parts:
        parts.append("<span>Market data unavailable</span>")
    parts.append(f'<span>ET {now_et.strftime("%H:%M:%S")}</span>')
    ticker_html = '<span class="sep">•</span>'.join(parts)
    st.markdown(f'<div class="ticker-wrap"><div class="ticker">{ticker_html}</div></div>', unsafe_allow_html=True)


def render_quote_card(label: str, ticker: str):
    quote = quick_quote(ticker)
    symbol = clean_ticker(ticker)
    with st.container():
        if quote is None:
            st.markdown(
                f'<div class="card"><div class="metric-label">{label} · {symbol}</div>'
                f'<div class="metric-value">—</div>'
                f'<div class="metric-delta">no data</div></div>',
                unsafe_allow_html=True,
            )
            return
        price, chg, pct = quote
        cls = "up" if chg >= 0 else "down"
        arrow = "▲" if chg >= 0 else "▼"
        st.markdown(
            f'<div class="card"><div class="metric-label">{label} · {symbol}</div>'
            f'<div class="metric-value">${price:,.2f}</div>'
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
        instructions=[
            "Please analyze the user's financial query.",
            "Focus on US-listed stocks (NYSE/NASDAQ) and report figures in USD.",
        ],
    )
    team_manager = Team(
        name="Financial Advisor",
        model=Groq(id=MODEL_ID),
        members=[web_agent, finance_agent],
        instructions=[
    "Combine web research and financial data to produce educational market analysis.",
    "Focus exclusively on US-listed stocks (NYSE/NASDAQ).",
    "Explain technical indicators, trends, and market context for learning.",
    "Avoid giving personalized investment recommendations.",
    "Clearly state that the analysis is for educational purposes only.",
        ],
    )
    return team_manager


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown('<div class="brand-eyebrow">Learning Configuration</div>', unsafe_allow_html=True)
    stock_a = st.text_input("Stock A", value="AAPL")
    stock_b = st.text_input("Stock B (optional)", value="MSFT")
    trade_style = st.selectbox("Trade style", ["swing trade", "intraday", "long-term hold"])

    st.markdown("---")
    st.markdown('<div class="brand-eyebrow">Stack</div>', unsafe_allow_html=True)
    st.caption(f"Model — Groq `{MODEL_ID}`")
    st.caption("Research — DuckDuckGo")
    st.caption("Market data — yfinance (NYSE/NASDAQ)")

    st.markdown("---")
    st.markdown('<div class="brand-eyebrow">Session history</div>', unsafe_allow_html=True)
    if st.session_state.history:
        for q in reversed(st.session_state.history[-8:]):
            st.markdown(f'<span class="session-pill">{q}</span>', unsafe_allow_html=True)
    else:
        st.caption("No queries yet this session.")

# ---------------------------------------------------------------------------
# Header with DecodixAI Logo (upper right corner)
# ---------------------------------------------------------------------------

logo_url = "https://raw.githubusercontent.com/learnermp09/swing_desk/main/decodixAI.png"
logo_html = f'<img src="{logo_url}" alt="DecodixAI Logo" style="max-height: 70px; width: auto; display: block;">'

st.markdown(
    f"""
<div class="header-container">
    <div class="header-left">
        <div class="brand-eyebrow-header">DecodixAI</div>
        <div class="brand-title-header">Swing Desk</div>
        <div class="brand-sub-header">Educational AI Trading Analysis</div>
    </div>
    <div class="header-right">
        {logo_html}
    </div>
</div>
    """,
    unsafe_allow_html=True
)
# ---------------------------------------------------------------------------
# Header + live ticker
# ---------------------------------------------------------------------------
render_ticker()

st.markdown(
    """
<div class="brand-title">
US Stock Swing Trading Learning Assistant
</div>

<div class="brand-sub">
Educational AI analysis using live NYSE/NASDAQ market data and web research.
Designed for learning market analysis—not for executing real trades.
</div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Quick quote cards (fast, independent of the LLM call below)
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

# Display patience message below submit button
st.markdown(
    '<div class="patience-message">⏳ Agents can take around 2 mins for proper research... keep patience!</div>',
    unsafe_allow_html=True
)

if run and query.strip():
    team_manager = build_team()
    with st.spinner("web_agent and finance_agent are on it..."):
        try:
            response = team_manager.run(query)
            content = response.content
            st.session_state.history.append(query.strip())
            st.markdown(f'<div class="analysis-panel">{content}</div>', unsafe_allow_html=True)
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            # Check if it's a rate limit error
            if "rate_limit_exceeded" in error_msg.lower() or "rate limit reached" in error_msg.lower():
                # Extract time from error message
                time_match = re.search(r'please try again in (\d+)m([\d.]+)s', error_msg)
                if time_match:
                    minutes = time_match.group(1)
                    seconds = time_match.group(2)
                    time_display = f"{minutes}m {seconds}s"
                else:
                    # Fallback: try to extract any time pattern
                    time_match = re.search(r'in (\d+[m.]?\d*s?)', error_msg)
                    if time_match:
                        time_display = time_match.group(1)
                    else:
                        time_display = "a few minutes"
                
                st.markdown(
                    f"""
                    <div class="rate-limit-error">
                        <strong>⏰ Tokens Expired - Rate Limit Reached</strong><br><br>
                        The API rate limit has been exceeded. Please wait <span class="timer">{time_display}</span> before trying again.<br><br>
                        <small>💡 Need more tokens? Upgrade to Dev Tier at <a href="https://console.groq.com/settings/billing" target="_blank">Groq Console</a></small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # Handle other errors
                st.error(f"An error occurred: {error_msg}")

st.write("")
st.warning(
    """
    **Educational Use Only**

    This application is intended exclusively for learning US stock market
    analysis and swing-trading concepts.

    • It is NOT financial or investment advice.
    • It does NOT recommend buying or selling securities.
    • Market data may be delayed or incomplete.
    • Always perform your own research and consult a qualified financial advisor
      before making investment decisions.
    """
)