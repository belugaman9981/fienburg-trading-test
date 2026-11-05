"""
trader_lab.py - train a long/short model on web-sourced price data and
benchmark it against standard strategies.

Usage:
    pip install -r requirements.txt
    python trader_lab.py --ticker SPY --start 2010-01-01
    python trader_lab.py --ticker BTC-USD --cost-bps 10
    python trader_lab.py --csv mydata.csv          # needs Date, Close (Volume optional)
    python trader_lab.py --synthetic               # offline smoke test
    python trader_lab.py --ticker SPY --signal     # also print the model's call for the next bar

Data sources (tried in order): yfinance (Yahoo Finance) -> Stooq CSV endpoint
-> local cache from a previous successful download (data_cache/).

How it works
  1. Download daily OHLCV from the web (and cache it).
  2. Build features (returns, volatility, RSI, SMA ratios, volume change).
  3. Walk-forward: retrain on all past data every N days, predict the next
     block only. The model never sees the future.
  4. Convert P(up) into LONG / SHORT / FLAT, charge transaction costs.
  5. Compare to: buy & hold, always short, SMA crossover, 6-month momentum,
     and a distribution of random long/short traders.

Outputs (written to --out, default "."): equity_curves.png, results.csv,
oos_returns.csv.

Research tool only. Not financial advice.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DEFAULT_PPY = 252  # trading periods per year, used only when it can't be inferred


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def synthetic_data(n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # slight momentum + vol clustering so there is *something* to learn
    r = np.zeros(n)
    vol = 0.01
    for i in range(1, n):
        vol = np.sqrt(0.000005 + 0.08 * r[i - 1] ** 2 + 0.9 * vol ** 2)
        r[i] = 0.0004 + 0.05 * r[i - 1] + vol * rng.standard_normal()
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2012-01-01", periods=n)
    return pd.DataFrame({"Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n)}, index=idx)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns, fix column case, keep Close (+Volume)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: str(c).strip().title() for c in df.columns})
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    if "Close" not in df.columns:
        raise ValueError(f"no Close column found (columns: {list(df.columns)})")
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    cols = [c for c in ("Close", "Volume") if c in df.columns]
    df = df[cols].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["Close"])


def cache_path(ticker: str, cache_dir: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in ticker.upper())
    return os.path.join(cache_dir, f"{safe}.csv")


def load_data(args) -> pd.DataFrame:
    if args.synthetic:
        return synthetic_data()

    if args.csv:
        df = pd.read_csv(args.csv)
        date_col = next((c for c in df.columns if str(c).strip().lower() in ("date", "datetime", "time")), df.columns[0])
        df = df.set_index(date_col)
        return normalize_frame(df)

    errors = []

    try:
        import yfinance as yf
        df = yf.download(args.ticker, start=args.start, progress=False, auto_adjust=True)
        if len(df):
            return _cache_and_return(normalize_frame(df), args)
        errors.append("yfinance returned no rows")
    except Exception as e:  # network, missing package, Yahoo schema change...
        errors.append(f"yfinance: {e}")
        print(f"[yfinance failed: {e}] trying Stooq...", file=sys.stderr)

    try:
        sym = args.ticker.lower()
        sym = sym if "." in sym else sym + ".us"
        df = pd.read_csv(f"https://stooq.com/q/d/l/?s={sym}&i=d", index_col="Date")
        df = normalize_frame(df).loc[args.start:]
        if len(df):
            return _cache_and_return(df, args)
        errors.append("stooq returned no rows")
    except Exception as e:
        errors.append(f"stooq: {e}")
        print(f"[stooq failed: {e}] trying local cache...", file=sys.stderr)

    path = cache_path(args.ticker, args.cache_dir)
    if os.path.exists(path):
        print(f"[using cached data: {path}]", file=sys.stderr)
        return normalize_frame(pd.read_csv(path, index_col=0)).loc[args.start:]

    sys.exit("Could not download data (" + "; ".join(errors) + ").\n"
             "Check your connection, or download a CSV yourself and run with --csv file.csv "
             "(needs Date and Close columns), or try --synthetic for an offline test.")


def _cache_and_return(df: pd.DataFrame, args) -> pd.DataFrame:
    try:
        os.makedirs(args.cache_dir, exist_ok=True)
        df.to_csv(cache_path(args.ticker, args.cache_dir))
    except OSError:
        pass  # caching is best-effort
    return df


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Observations per calendar year (~252 for stocks, ~365 for crypto)."""
    if len(index) < 3:
        return DEFAULT_PPY
    days = (index[-1] - index[0]).days
    return DEFAULT_PPY if days <= 0 else 365.25 * (len(index) - 1) / days


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
    if "Volume" in df and df["Volume"].notna().any() and (df["Volume"] > 0).any():
        f["vol_chg"] = np.log((df["Volume"] + 1) / (df["Volume"].rolling(20).mean() + 1))
    return f


# --------------------------------------------------------------------------
# Walk-forward model
# --------------------------------------------------------------------------
def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150,
        l2_regularization=1.0, min_samples_leaf=40, random_state=0,
    )


def walk_forward_probs(X: pd.DataFrame, y: pd.Series, min_train=750, step=63) -> pd.Series:
    """P(next-day up) using only data available at each point in time."""
    probs = pd.Series(np.nan, index=X.index)
    i = min_train
    while i < len(X):
        j = min(i + step, len(X))
        model = make_model()
        # drop the last row of the train window: its label uses day i's return
        Xt, yt = X.iloc[: i - 1], y.iloc[: i - 1]
        if yt.nunique() < 2:  # degenerate window; skip rather than crash
            i = j
            continue
        model.fit(Xt, yt)
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


def latest_signal(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, thresh: float):
    """Fit on every labelled row and score the newest bar. Returns (date, P(up), position)."""
    feats = make_features(df).dropna()
    if feats.empty or len(X) < 100 or y.nunique() < 2:
        return None
    x_last = feats.iloc[[-1]]
    model = make_model().fit(X, y)
    p = float(model.predict_proba(x_last[X.columns])[0, 1])
    pos = 1 if p > 0.5 + thresh else -1 if p < 0.5 - thresh else 0
    return x_last.index[0], p, pos


# --------------------------------------------------------------------------
# Backtest + metrics
# --------------------------------------------------------------------------
def backtest(pos: pd.Series, fwd_ret: pd.Series, cost_bps: float) -> pd.Series:
    """pos[t] is decided at close t, earns fwd_ret[t] (= return t -> t+1)."""
    pos = pos.reindex(fwd_ret.index)
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])  # cost of entering the first position
    return pos * fwd_ret - turnover * cost_bps / 1e4


def sharpe(r: pd.Series, ppy: float) -> float:
    r = r.dropna()
    return float(np.sqrt(ppy) * r.mean() / (r.std() + 1e-12))


def metrics(r: pd.Series, pos: pd.Series = None, ppy: float = DEFAULT_PPY) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / ppy
    out = {
        "Total return %": 100 * (eq.iloc[-1] - 1),
        "CAGR %": 100 * (eq.iloc[-1] ** (1 / years) - 1) if years > 0 and eq.iloc[-1] > 0 else np.nan,
        "Sharpe": sharpe(r, ppy),
        "Max DD %": 100 * (eq / eq.cummax() - 1).min(),
        "Hit rate %": 100 * (r[r != 0] > 0).mean() if (r != 0).any() else np.nan,
    }
    if pos is not None:
        out["Trades"] = int((pos.diff().abs() > 0).sum())
        out["Time long %"] = 100 * (pos == 1).mean()
        out["Time short %"] = 100 * (pos == -1).mean()
    return out


def random_baseline(fwd_ret: pd.Series, cost_bps: float, n=300, hold=5, seed=1, ppy: float = DEFAULT_PPY):
    """Sharpe distribution of random long/short traders (same cost model)."""
    rng = np.random.default_rng(seed)
    sharpes = []
    L = len(fwd_ret)
    for _ in range(n):
        blocks = rng.choice([-1.0, 1.0], size=L // hold + 1)
        pos = pd.Series(np.repeat(blocks, hold)[:L], index=fwd_ret.index)
        sharpes.append(sharpe(backtest(pos, fwd_ret, cost_bps), ppy))
    return np.array(sharpes)


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Walk-forward ML long/short backtester")
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="cost per unit of position change")
    ap.add_argument("--thresh", type=float, default=0.02, help="confidence band around 0.5; inside = flat")
    ap.add_argument("--min-train", type=int, default=750)
    ap.add_argument("--step", type=int, default=63, help="retrain every N days")
    ap.add_argument("--signal", action="store_true", help="also print the model's call for the next bar")
    ap.add_argument("--out", default=".", help="folder for equity_curves.png / results.csv / oos_returns.csv")
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    df = load_data(args)
    if len(df) == 0:
        sys.exit("Dataset is empty.")
    print(f"Loaded {len(df)} rows: {df.index[0].date()} -> {df.index[-1].date()}")
    ppy = periods_per_year(df.index)

    close_full = df["Close"]
    fwd = close_full.pct_change().shift(-1)       # return from t to t+1
    X_all = make_features(df)
    y_all = (fwd > 0).astype(int)

    valid = X_all.dropna().index.intersection(fwd.dropna().index)
    X, y, fwd, close = X_all.loc[valid], y_all.loc[valid], fwd.loc[valid], close_full.loc[valid]

    need = args.min_train + args.step
    if len(X) < need:
        sys.exit(f"Only {len(X)} usable rows after building features; need at least {need} "
                 f"(--min-train {args.min_train} + --step {args.step}). "
                 "Use an earlier --start, a smaller --min-train, or more data.")

    probs = walk_forward_probs(X, y, args.min_train, args.step)
    oos = probs.dropna().index                   # strictly out-of-sample period
    if len(oos) == 0:
        sys.exit("The model produced no out-of-sample predictions (training windows had a single class).")
    probs, fwd_o = probs[oos], fwd[oos]

    acc = ((probs > 0.5).astype(int) == y[oos]).mean()
    print(f"Out-of-sample directional accuracy: {100*acc:.2f}%  "
          f"(always-up baseline: {100*y[oos].mean():.2f}%)")

    # Strategies. Moving averages come from the full history so they are not
    # re-warmed-up by the feature-NaN trimming above.
    strategies = {}
    strategies["ML long/short (flat when unsure)"] = probs_to_position(probs, args.thresh)
    strategies["ML long/short (always in)"] = probs_to_position(probs, 0, allow_flat=False)
    strategies["Buy & hold"] = pd.Series(1.0, index=oos)
    strategies["Always short"] = pd.Series(-1.0, index=oos)

    sma_fast, sma_slow = close_full.rolling(50).mean(), close_full.rolling(200).mean()
    strategies["SMA 50/200 long/short"] = pd.Series(
        np.where(sma_fast > sma_slow, 1.0, -1.0), index=close_full.index).loc[oos]
    mom = close_full.pct_change(126)
    strategies["6-mo momentum long/short"] = pd.Series(
        np.where(mom > 0, 1.0, -1.0), index=close_full.index).loc[oos]

    rows, rets = {}, {}
    for name, pos in strategies.items():
        r = backtest(pos, fwd_o, args.cost_bps)
        rets[name] = r
        rows[name] = metrics(r, pos, ppy)
    table = pd.DataFrame(rows).T.round(2)
    print("\n=== Out-of-sample results (after costs) ===")
    print(table.to_string())

    # Is the ML Sharpe distinguishable from luck?
    rnd = random_baseline(fwd_o, args.cost_bps, ppy=ppy)
    ml_name = "ML long/short (flat when unsure)"
    ml_sharpe = table.loc[ml_name, "Sharpe"]
    pct = 100 * (rnd < ml_sharpe).mean()
    print(f"\nRandom long/short traders: Sharpe mean {rnd.mean():.2f}, "
          f"95th pct {np.percentile(rnd, 95):.2f}")
    print(f"ML Sharpe {ml_sharpe:.2f} beats {pct:.0f}% of random traders.")
    bh = table.loc["Buy & hold", "Sharpe"]
    if ml_sharpe <= bh:
        print("-> It does NOT beat buy & hold on risk-adjusted terms here.")
    if pct < 95:
        print("-> Not statistically distinguishable from luck (needs >95th percentile).")

    if args.signal:
        sig = latest_signal(df, X, y, args.thresh)
        if sig is None:
            print("\nNot enough data for a latest signal.")
        else:
            d, p, pos = sig
            label = {1: "LONG", -1: "SHORT", 0: "FLAT"}[pos]
            print(f"\nLatest signal (bar {d.date()}): P(up) = {p:.3f} -> {label}")
            print("(Research output, not advice. See the statistics above for how much it is worth.)")

    try:
        os.makedirs(args.out, exist_ok=True)
        table.to_csv(os.path.join(args.out, "results.csv"))
        out = pd.DataFrame({"P_up": probs, "fwd_ret": fwd_o})
        for name, r in rets.items():
            out[name] = r
        out.to_csv(os.path.join(args.out, "oos_returns.csv"))
        print(f"\nSaved results.csv and oos_returns.csv to {os.path.abspath(args.out)}")
    except OSError as e:
        print(f"\n[could not write CSV outputs: {e}]", file=sys.stderr)

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for name, r in rets.items():
            if name == "Always short":
                continue
            ax.plot((1 + r.fillna(0)).cumprod(), label=name, lw=2.2 if name == ml_name else 1.2)
        ax.set_yscale("log")
        ax.set_title(f"Out-of-sample equity curves (cost {args.cost_bps} bps)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(args.out, "equity_curves.png")
        fig.savefig(path, dpi=130)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
