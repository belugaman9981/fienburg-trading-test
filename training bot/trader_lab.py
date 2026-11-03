"""
trader_lab.py - train a long/short model on web-sourced price data and
benchmark it against standard strategies.

Usage:
    pip install pandas numpy scikit-learn matplotlib yfinance
    python trader_lab.py --ticker SPY --start 2010-01-01
    python trader_lab.py --ticker BTC-USD --cost-bps 10
    python trader_lab.py --csv mydata.csv          # needs Date, Close (Volume optional)
    python trader_lab.py --synthetic               # offline smoke test

Data sources (tried in order): yfinance (Yahoo Finance) -> Stooq CSV endpoint.

How it works
  1. Download daily OHLCV from the web.
  2. Build features (returns, volatility, RSI, SMA ratios, volume change).
  3. Walk-forward: retrain on all past data every N days, predict the next
     block only. The model never sees the future.
  4. Convert P(up) into LONG / SHORT / FLAT, charge transaction costs.
  5. Compare to: buy & hold, always short, SMA crossover, 6-month momentum,
     and a distribution of random long/short traders.

Research tool only. Not financial advice.
"""
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(args) -> pd.DataFrame:
    if args.synthetic:
        rng = np.random.default_rng(0)
        n = 3000
        # slight momentum + vol clustering so there is *something* to learn
        r = np.zeros(n)
        vol = 0.01
        for i in range(1, n):
            vol = np.sqrt(0.000005 + 0.08 * r[i - 1] ** 2 + 0.9 * vol ** 2)
            r[i] = 0.0004 + 0.05 * r[i - 1] + vol * rng.standard_normal()
        close = 100 * np.exp(np.cumsum(r))
        idx = pd.bdate_range("2012-01-01", periods=n)
        return pd.DataFrame({"Close": close, "Volume": rng.integers(1e6, 5e6, n)}, index=idx)

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["Date"], index_col="Date").sort_index()
        return df[[c for c in ["Close", "Volume"] if c in df.columns]].dropna(subset=["Close"])

    try:
        import yfinance as yf
        df = yf.download(args.ticker, start=args.start, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df):
            return df[["Close", "Volume"]].dropna(subset=["Close"])
    except Exception as e:
        print(f"[yfinance failed: {e}] trying Stooq...", file=sys.stderr)

    sym = args.ticker.lower()
    sym = sym if "." in sym else sym + ".us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    df = pd.read_csv(url, parse_dates=["Date"], index_col="Date").sort_index()
    df = df.loc[args.start:]
    if df.empty:
        sys.exit("No data returned. Try --csv with a file you downloaded.")
    return df[["Close", "Volume"]].dropna(subset=["Close"])


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def rsi(close: pd.Series, n=14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    r = c.pct_change()
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 5, 10, 20, 60):
        f[f"ret_{k}"] = c.pct_change(k)
    f["vol_10"] = r.rolling(10).std()
    f["vol_20"] = r.rolling(20).std()
    f["vol_ratio"] = f["vol_10"] / (f["vol_20"] + 1e-12)
    f["rsi_14"] = rsi(c)
    f["sma_10_50"] = c.rolling(10).mean() / c.rolling(50).mean() - 1
    f["sma_50_200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    f["dist_high_60"] = c / c.rolling(60).max() - 1
    if "Volume" in df and df["Volume"].notna().any():
        f["vol_chg"] = np.log((df["Volume"] + 1) / (df["Volume"].rolling(20).mean() + 1))
    return f


# --------------------------------------------------------------------------
# Walk-forward model
# --------------------------------------------------------------------------
def walk_forward_probs(X: pd.DataFrame, y: pd.Series, min_train=750, step=63) -> pd.Series:
    """P(next-day up) using only data available at each point in time."""
    probs = pd.Series(np.nan, index=X.index)
    i = min_train
    while i < len(X):
        j = min(i + step, len(X))
        model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150,
            l2_regularization=1.0, min_samples_leaf=40, random_state=0,
        )
        # drop the last row of the train window: its label uses day i's return
        model.fit(X.iloc[: i - 1], y.iloc[: i - 1])
        probs.iloc[i:j] = model.predict_proba(X.iloc[i:j])[:, 1]
        i = j
    return probs


def probs_to_position(p: pd.Series, thresh: float, allow_flat=True) -> pd.Series:
    pos = pd.Series(0.0, index=p.index)
    pos[p > 0.5 + thresh] = 1.0
    pos[p < 0.5 - thresh] = -1.0
    if not allow_flat:
        pos[p >= 0.5] = 1.0
        pos[p < 0.5] = -1.0
    pos[p.isna()] = np.nan
    return pos


# --------------------------------------------------------------------------
# Backtest + metrics
# --------------------------------------------------------------------------
def backtest(pos: pd.Series, fwd_ret: pd.Series, cost_bps: float) -> pd.Series:
    """pos[t] is decided at close t, earns fwd_ret[t] (= return t -> t+1)."""
    turnover = pos.diff().abs().fillna(abs(pos.iloc[0]))
    return pos * fwd_ret - turnover * cost_bps / 1e4


def metrics(r: pd.Series, pos: pd.Series = None) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    out = {
        "Total return %": 100 * (eq.iloc[-1] - 1),
        "CAGR %": 100 * (eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
        "Sharpe": np.sqrt(TRADING_DAYS) * r.mean() / (r.std() + 1e-12),
        "Max DD %": 100 * (eq / eq.cummax() - 1).min(),
        "Hit rate %": 100 * (r[r != 0] > 0).mean() if (r != 0).any() else np.nan,
    }
    if pos is not None:
        out["Trades"] = int((pos.diff().abs() > 0).sum())
        out["Time long %"] = 100 * (pos == 1).mean()
        out["Time short %"] = 100 * (pos == -1).mean()
    return out


def random_baseline(fwd_ret: pd.Series, cost_bps: float, n=300, hold=5, seed=1):
    """Sharpe distribution of random long/short traders (same cost model)."""
    rng = np.random.default_rng(seed)
    sharpes = []
    L = len(fwd_ret)
    for _ in range(n):
        blocks = rng.choice([-1.0, 1.0], size=L // hold + 1)
        pos = pd.Series(np.repeat(blocks, hold)[:L], index=fwd_ret.index)
        r = backtest(pos, fwd_ret, cost_bps)
        sharpes.append(np.sqrt(TRADING_DAYS) * r.mean() / (r.std() + 1e-12))
    return np.array(sharpes)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="cost per unit of position change")
    ap.add_argument("--thresh", type=float, default=0.02, help="confidence band around 0.5; inside = flat")
    ap.add_argument("--min-train", type=int, default=750)
    ap.add_argument("--step", type=int, default=63, help="retrain every N days")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    df = load_data(args)
    print(f"Loaded {len(df)} rows: {df.index[0].date()} -> {df.index[-1].date()}")

    close = df["Close"]
    fwd = close.pct_change().shift(-1)           # return from t to t+1
    X = make_features(df)
    y = (fwd > 0).astype(int)

    valid = X.dropna().index.intersection(fwd.dropna().index)
    X, y, fwd, close = X.loc[valid], y.loc[valid], fwd.loc[valid], close.loc[valid]

    probs = walk_forward_probs(X, y, args.min_train, args.step)
    oos = probs.dropna().index                   # strictly out-of-sample period
    probs, fwd_o, close_o = probs[oos], fwd[oos], close[oos]

    # Out-of-sample classification accuracy
    acc = ((probs > 0.5).astype(int) == y[oos]).mean()
    print(f"Out-of-sample directional accuracy: {100*acc:.2f}%  "
          f"(always-up baseline: {100*y[oos].mean():.2f}%)")

    # Strategies
    strategies = {}
    strategies["ML long/short (flat when unsure)"] = probs_to_position(probs, args.thresh)
    strategies["ML long/short (always in)"] = probs_to_position(probs, 0, allow_flat=False)
    strategies["Buy & hold"] = pd.Series(1.0, index=oos)
    strategies["Always short"] = pd.Series(-1.0, index=oos)

    sma_fast, sma_slow = close.rolling(50).mean(), close.rolling(200).mean()
    strategies["SMA 50/200 long/short"] = pd.Series(
        np.where(sma_fast > sma_slow, 1.0, -1.0), index=close.index)[oos]
    mom = close.pct_change(126)
    strategies["6-mo momentum long/short"] = pd.Series(
        np.where(mom > 0, 1.0, -1.0), index=close.index)[oos]

    rows, rets = {}, {}
    for name, pos in strategies.items():
        r = backtest(pos, fwd_o, args.cost_bps)
        rets[name] = r
        rows[name] = metrics(r, pos)
    table = pd.DataFrame(rows).T.round(2)
    print("\n=== Out-of-sample results (after costs) ===")
    print(table.to_string())

    # Is the ML Sharpe distinguishable from luck?
    rnd = random_baseline(fwd_o, args.cost_bps)
    ml_sharpe = table.loc["ML long/short (flat when unsure)", "Sharpe"]
    pct = 100 * (rnd < ml_sharpe).mean()
    print(f"\nRandom long/short traders: Sharpe mean {rnd.mean():.2f}, "
          f"95th pct {np.percentile(rnd, 95):.2f}")
    print(f"ML Sharpe {ml_sharpe:.2f} beats {pct:.0f}% of random traders.")
    bh = table.loc["Buy & hold", "Sharpe"]
    if ml_sharpe <= bh:
        print("-> It does NOT beat buy & hold on risk-adjusted terms here.")
    if pct < 95:
        print("-> Not statistically distinguishable from luck (needs >95th percentile).")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for name, r in rets.items():
            if name in ("Always short",):
                continue
            ax.plot((1 + r).cumprod(), label=name, lw=2.2 if name.startswith("ML long/short (flat") else 1.2)
        ax.set_yscale("log")
        ax.set_title(f"Out-of-sample equity curves (cost {args.cost_bps} bps)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("equity_curves.png", dpi=130)
        print("\nSaved equity_curves.png")


if __name__ == "__main__":
    main()