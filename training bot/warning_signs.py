"""
warning_signs.py  -  early-warning candle signals for the training bot.

Python port of the "Time machine" detectors in PaperTrade, so the bot sees the
same things you see on the chart: John Wick (long upper wick) tops, hammers,
bearish engulfing, shrinking candles, price rising on fading volume, heavy
selling days, losing the 50-day average, death cross and overheated RSI.

Every signal at row t uses ONLY data up to row t (no look-ahead), so the columns
are safe to use as model features. add_targets()/scorecard() DO look ahead on
purpose, so keep their columns out of your features.

Use inside trader_lab.py:
    from warning_signs import add_features, FEATURE_COLUMNS
    df = add_features(df)                      # df has open/high/low/close/volume
    X = df[FEATURE_COLUMNS].astype(float).fillna(0)

Run on its own for a quick report:
    py warning_signs.py AAPL 3        (ticker, years; needs yfinance)
"""
import sys
import numpy as np
import pandas as pd

# How much each signal adds to the danger score (negative = good sign).
WEIGHTS = {
    "sig_wick_top": 20, "sig_engulf": 15, "sig_slow": 15, "sig_voldiv": 15,
    "sig_dist": 20, "sig_brk50": 20, "sig_death": 15, "sig_hot": 10,
    "sig_wick_bot": -15,
}
SIGNAL_NAMES = {
    "sig_wick_top": "John Wick top", "sig_engulf": "Bearish engulfing",
    "sig_slow": "Momentum slowing", "sig_voldiv": "Rising on fading volume",
    "sig_dist": "Heavy selling day", "sig_brk50": "Lost 50-day average",
    "sig_death": "Death cross", "sig_hot": "Overheated (RSI>75)",
    "sig_wick_bot": "Hammer (long lower wick)",
}
FEATURE_COLUMNS = list(WEIGHTS) + [
    "upper_wick_ratio", "lower_wick_ratio", "body_ratio",
    "dist_ma20", "dist_ma50", "rsi14", "vol_ratio", "danger",
]


def _clean(df):
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):          # yfinance returns this
        d.columns = d.columns.get_level_values(0)
    d.columns = [str(c).lower() for c in d.columns]
    for col in ("open", "high", "low", "close", "volume"):
        if col not in d.columns:
            raise ValueError("missing column: " + col)
    return d


def _cooldown(s, gap):
    """Keep a signal only if the last kept one was at least `gap` rows ago."""
    out = np.zeros(len(s), dtype=bool)
    last = -10 ** 9
    for i, x in enumerate(s.to_numpy()):
        if x and i - last >= gap:
            out[i] = True
            last = i
    return pd.Series(out, index=s.index)


def add_features(df, cooldown=6):
    d = _clean(df)
    o, h, l, c, v = (d[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
    v20 = v.rolling(20).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["rsi14"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(100).where(gain.notna())

    rallying = (c > ma20) & (c / c.shift(5) > 1.02)
    falling = (c < ma20) & (c / c.shift(5) < 0.98)
    pc, po = c.shift(), o.shift()

    d["sig_wick_top"] = rallying & (upper >= 2 * body) & (upper >= 0.55 * rng)
    d["sig_wick_bot"] = falling & (lower >= 2 * body) & (lower >= 0.55 * rng)
    d["sig_engulf"] = (pc > po) & (c < o) & (o >= pc) & (c <= po) & (pc > ma20.shift())
    d["sig_slow"] = (body.rolling(5).mean() < 0.5 * body.shift(5).rolling(10).mean()) & (c > ma20 * 1.03)
    d["sig_voldiv"] = (c >= c.shift(1).rolling(20).max()) & (v.rolling(3).mean() < 0.75 * v20)
    d["sig_dist"] = (c < o) & (v > 2 * v20) & (((c - l) / rng) < 0.25)
    d["sig_brk50"] = (c < ma50) & (c.shift() >= ma50.shift())
    d["sig_death"] = (ma20 < ma50) & (ma20.shift() >= ma50.shift())
    d["sig_hot"] = (d["rsi14"] > 75) & (d["rsi14"].shift() <= 75)

    for k in WEIGHTS:
        d[k] = d[k].fillna(False).astype(bool)
    d.iloc[:55, [d.columns.get_loc(k) for k in WEIGHTS]] = False   # warm-up rows
    for k in ("sig_slow", "sig_voldiv", "sig_hot"):
        d[k] = _cooldown(d[k], cooldown)

    # continuous features for the model
    d["upper_wick_ratio"] = (upper / rng).fillna(0)
    d["lower_wick_ratio"] = (lower / rng).fillna(0)
    d["body_ratio"] = (body / rng).fillna(0)
    d["dist_ma20"] = c / ma20 - 1
    d["dist_ma50"] = c / ma50 - 1
    d["vol_ratio"] = v / v20

    # danger score 0-100: recent signs, fading out over 10 rows
    w = sum(d[k].astype(float) * wt for k, wt in WEIGHTS.items())
    d["danger"] = sum(w.shift(a).fillna(0) * (1 - a / 10) for a in range(10)).clip(0, 100)
    return d


def add_targets(d, horizon=5):
    """LOOKS AHEAD. For labels and scorecards only, never as features."""
    d = d.copy()
    d["fwd_ret"] = d["close"].shift(-horizon) / d["close"] - 1
    d["fell"] = (d["fwd_ret"] < 0).astype(float).where(d["fwd_ret"].notna())
    return d


def scorecard(d, horizon=5):
    """How often price was lower `horizon` rows after each signal vs a random row."""
    t = add_targets(d, horizon).iloc[55:].dropna(subset=["fwd_ret"])
    rows = [("Any random day", len(t), t["fell"].mean(), t["fwd_ret"].mean())]
    for k in WEIGHTS:
        s = t[t[k]]
        rows.append((SIGNAL_NAMES[k], len(s),
                     s["fell"].mean() if len(s) else np.nan,
                     s["fwd_ret"].mean() if len(s) else np.nan))
    return pd.DataFrame(rows, columns=["signal", "times", "fell_pct", "avg_move"])


if __name__ == "__main__":
    import yfinance as yf
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    yrs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    raw = yf.download(sym, period=f"{yrs}y", progress=False, auto_adjust=True)
    d = add_features(raw)
    print(f"\n{sym}: {len(d)} daily candles\n")
    sc = scorecard(d)
    sc["fell_pct"] = (sc["fell_pct"] * 100).round(0)
    sc["avg_move"] = (sc["avg_move"] * 100).round(2)
    print(sc.to_string(index=False))
    last = d.iloc[-1]
    on = [SIGNAL_NAMES[k] for k in WEIGHTS if last[k]]
    print(f"\nLatest danger score: {last['danger']:.0f}/100   RSI: {last['rsi14']:.0f}")
    print("Signals on the latest candle:", ", ".join(on) if on else "none")
    print("\nCompare each row to 'Any random day'. Rows with few samples mean little.")
 
