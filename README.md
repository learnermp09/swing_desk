# swing_desk
Multi-agent AI financial advisor for US stock analysis built with Streamlit, Agno, Groq, DuckDuckGo, and yfinance

# Swing Desk — US Equity Analyst

## [Disclaimer: Just for learning purpose and not for real trade]

A small Streamlit app that helps get a quick second opinion before a swing trade on US-listed stocks (NYSE/NASDAQ). Give it two tickers, it pulls live price data and runs a couple of AI agents to research and compare them.

Built this mostly for showcase purpose — comparing stocks before one enters a trade usually meant switching between five browser tabs. This puts the important bits in one place.

## What it does

- Pulls live NYSE/NASDAQ quotes (via yfinance) for two stocks you pick
- Runs a small team of agents (one does web research, one pulls financial data) to compare them for a swing trade
- Shows a scrolling ticker for the S&P 500, Dow Jones and Nasdaq at the top
- Keeps a short history of what you've asked in the sidebar, for the current session

It is not a stock tip generator. Treat the output as a starting point for your own research, not advice.

## How it's built

- **Streamlit** for the UI
- **Agno** for the agent framework (a web-search agent + a finance-data agent, coordinated by a team leader)
- **Groq** for the LLM (currently running Qwen 3.6 27B — cheap and fast)
- **yfinance** for stock and index prices
- **DuckDuckGo search** for the web-research side

## Project structure

```
swing_desk/
├── financial_advisor_app.py   # the whole app — UI, agents, market data calls
├── requirements.txt           # Python dependencies
├── .gitignore                 # keeps .env and other local files out of git
├── LICENSE                    # MIT
└── README.md
```

Everything lives in one file (`financial_advisor_app.py`) on purpose — it's small enough that splitting it into modules would add more overhead than it saves.

## Running it locally

1. Clone the repo and install requirements:

   ```bash
   git clone https://github.com/learnermp09/swing_desk.git
   cd swing_desk
   pip install -r requirements.txt
   ```

2. Get a free API key from [console.groq.com](https://console.groq.com) and add it to a `.env` file in the project root:

   ```
   GROQ_API_KEY=your_key_here
   ```

3. Run it:

   ```bash
   streamlit run financial_advisor_app.py
   ```

## Deploying on Streamlit Community Cloud

1. Push this repo to your own GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Pick this repo, set the main file to `financial_advisor_app.py`, and deploy.
4. Before it'll run, add your Groq key under **App settings → Secrets**:

   ```
   GROQ_API_KEY = "your_key_here"
   ```

That's it — no other config needed.

## A few honest limitations

- Price data can lag the exchange by a few minutes
- The web-research agent uses DuckDuckGo, which occasionally returns nothing useful for obscure queries — it'll just work with less context if that happens.
- The market clock in the ticker is a rough ET estimate and doesn't adjust for daylight saving — good enough for a glance, not for anything time-critical.
- Small-cap and less-liquid stocks sometimes have patchy data on yfinance. Works best on well-covered large and mid-cap names.

## License

MIT — do whatever you want with it.
