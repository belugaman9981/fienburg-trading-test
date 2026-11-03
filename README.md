# fienburg trading test

Two small, independent research tools for practising trading ideas without risking money.

| Folder | What it is |
|---|---|
| [`platform/`](platform) | **PaperTrade**: a browser paper-trading simulator (market, limit and stop orders, candlestick charts, optional live prices from Alpaca). One static `index.html`, no backend. |
| [`training bot/`](training%20bot) | **trader_lab.py**: trains a walk-forward ML long/short model on daily prices and benchmarks it against buy & hold, SMA crossover, momentum and random traders. |

Both are practice and research tools, not financial advice.

## Quick start

**Paper trading app**
```bash
cd platform
python3 -m http.server 8000      # then open http://localhost:8000
```

**Backtester**
```bash
cd "training bot"
pip install -r requirements.txt
python trader_lab.py --synthetic                 # offline smoke test
python trader_lab.py --ticker SPY --signal       # real data + the model's call for the next bar
python trader_lab.py --ticker BTC-USD --cost-bps 10
python trader_lab.py --csv mydata.csv            # needs Date and Close columns
```
Outputs `equity_curves.png`, `results.csv` and `oos_returns.csv` (use `--out DIR` to choose where).
If Yahoo and Stooq are both unreachable it falls back to the last cached download in `data_cache/`.

## Tests
```bash
cd "training bot" && pytest -q          # backtest maths, no-lookahead check, CLI end to end
cd platform && npm install && npm test  # drives the real page headlessly, incl. mocked Alpaca live mode
```

## Reading the backtester honestly
The script prints whether the model beats buy & hold *and* whether its Sharpe ratio beats 95% of random
traders. Daily-direction models usually do not clear that bar after costs; treat a good-looking curve as a
reason to check for leakage and overfitting before anything else.
