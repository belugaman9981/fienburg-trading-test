# PaperTrade

A paper trading simulator that runs entirely in the browser. Start with $100,000 in fake cash, place market, limit and stop orders, and track your positions and portfolio P/L. No real money, no backend, no API keys.

**Live demo:** https://belugaman9981.github.io/papertrade/

## Features
- Market, limit and stop orders (limit orders fill when the price crosses your number; stop orders become market orders once the price trades through the stop; all can be cancelled, and open buys reserve buying power)
- Candlestick chart with MA(7) and MA(25) overlays, volume bars, high/low markers and a live last-price tag
- 12 stocks/ETFs plus BTC/USD (fractional quantities) in a live-updating watchlist with sparklines
- Positions table with average cost, P/L and one-click close
- Portfolio equity chart, buying power that accounts for open buy orders
- Speed selector (Normal, Fast, Turbo) for the simulated market
- Account saved in `localStorage`, with a reset button
- Dark mode by default with a light mode toggle, responsive layout

## Live prices (optional)
Click **⚡ Live prices**, paste the API keys from your free [Alpaca](https://alpaca.markets) Paper Trading account, and press Connect. The page then streams every trade from Alpaca’s free market data (IEX feed) over a WebSocket, with a REST snapshot every 2 seconds as a backup. The free plan allows one stream connection at a time, so keep a single tab open. Keys are stored only in your browser’s `localStorage` and are sent only to Alpaca. Never commit keys to the repo. Live and simulated modes keep separate paper accounts. Live prices only work when the page is served from your own site (GitHub Pages or localhost), and stock prices only move during US market hours. BTC/USD uses Alpaca’s crypto feed and trades 24/7.

## Run locally
It is a single static file. Open `index.html` in a browser, or serve the folder:

```bash
python3 -m http.server 8000
```

## Notes
Prices are **simulated** with a random walk, so they are not real market data. This is a practice tool, not financial advice, and it is not affiliated with Alpaca or any broker.

## Tests
Headless tests drive the real page in jsdom, including the Alpaca live mode against a mocked API and WebSocket (no keys or network needed):

```bash
npm install
npm test
```

## Ideas for next steps
- Stop-limit and trailing-stop orders
- Accounts and a shared leaderboard (needs a backend such as Supabase)

## License
MIT
