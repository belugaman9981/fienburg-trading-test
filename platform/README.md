# PaperTrade

A paper trading simulator that runs entirely in the browser. Start with $100,000 in fake cash, place market and limit orders, and track your positions and portfolio P/L. No real money, no backend, no API keys.

**Live demo:** https://belugaman9981.github.io/papertrade/

## Features
- Market and limit orders (limit orders fill when the price crosses your number, and can be cancelled)
- Candlestick chart with MA(7) and MA(25) overlays, volume bars, high/low markers and a live last-price tag
- 12 tickers with a live-updating watchlist and sparklines
- Positions table with average cost, P/L and one-click close
- Portfolio equity chart, buying power that accounts for open buy orders
- Speed selector (Normal, Fast, Turbo) for the simulated market
- Account saved in `localStorage`, with a reset button
- Dark mode by default with a light mode toggle, responsive layout

## Run locally
It is a single static file. Open `index.html` in a browser, or serve the folder:

```bash
python3 -m http.server 8000
```

## Notes
Prices are **simulated** with a random walk, so they are not real market data. This is a practice tool, not financial advice, and it is not affiliated with Alpaca or any broker.

## Ideas for next steps
- Real prices via a market data API (for example Alpaca’s free market data, proxied through a small backend so keys stay private)
- Stop and stop-limit orders
- Accounts and a shared leaderboard (needs a backend such as Supabase)

## License
MIT
