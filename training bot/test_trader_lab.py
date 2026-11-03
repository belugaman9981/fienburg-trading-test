"""Run with:  pytest -q   (from the 'training bot' folder)"""
import numpy as np
import pandas as pd
import pytest

import trader_lab as tl


@pytest.fixture(scope="module")
def data():
    df = tl.synthetic_data(n=1800, seed=3)
    fwd = df["Close"].pct_change().shift(-1)
    X = tl.make_features(df)
    y = (fwd > 0).astype(int)
    valid = X.dropna().index.intersection(fwd.dropna().index)
    return df, X.loc[valid], y.loc[valid], fwd.loc[valid]


def test_backtest_math():
    idx = pd.RangeIndex(4)
    pos = pd.Series([1.0, 1.0, -1.0, 0.0], index=idx)
    fwd = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx)
    r = tl.backtest(pos, fwd, cost_bps=10)
    # entry costs 10bp, the flip costs 2*10bp, going flat costs 10bp
    assert r.round(6).tolist() == [0.009, 0.02, -0.032, -0.001]


def test_metrics_buy_and_hold():
    r = pd.Series([0.01] * 252)
    m = tl.metrics(r, pd.Series(1.0, index=r.index), ppy=252)
    assert m["Total return %"] == pytest.approx(100 * (1.01 ** 252 - 1))
    assert m["Max DD %"] == 0
    assert m["Time long %"] == 100


def test_periods_per_year_detects_crypto_vs_stocks():
    assert tl.periods_per_year(pd.bdate_range("2020-01-01", periods=1000)) == pytest.approx(261, abs=3)
    assert tl.periods_per_year(pd.date_range("2020-01-01", periods=1000)) == pytest.approx(365, abs=2)


def test_probs_to_position_thresholds():
    p = pd.Series([0.60, 0.51, 0.40, np.nan])
    pos = tl.probs_to_position(p, 0.02)
    assert pos.iloc[:3].tolist() == [1.0, 0.0, -1.0] and np.isnan(pos.iloc[3])
    assert tl.probs_to_position(p.dropna(), 0, allow_flat=False).tolist() == [1.0, 1.0, -1.0]


def test_walk_forward_has_no_lookahead(data):
    """Rewriting the far future must not change any earlier prediction."""
    df, X, y, _ = data
    base = tl.walk_forward_probs(X, y, min_train=750, step=63)

    cut = 750 + 63 * 12                      # a block boundary, in model-row positions
    cut_date = X.index[cut]                  # (model rows start after indicator warm-up)
    df2 = df.copy()
    rng = np.random.default_rng(9)
    future = df2.index >= cut_date
    df2.loc[future, "Close"] *= np.exp(np.cumsum(rng.normal(0, 0.05, future.sum())))
    fwd2 = df2["Close"].pct_change().shift(-1)
    X2 = tl.make_features(df2)
    y2 = (fwd2 > 0).astype(int)
    valid2 = X2.dropna().index.intersection(fwd2.dropna().index)
    alt = tl.walk_forward_probs(X2.loc[valid2], y2.loc[valid2], min_train=750, step=63)

    a, b = base.iloc[:cut], alt.reindex(base.index).iloc[:cut]
    assert a.notna().sum() > 0
    pd.testing.assert_series_equal(a, b)


def test_cli_end_to_end(tmp_path, capsys):
    tl.main(["--synthetic", "--signal", "--out", str(tmp_path), "--no-plot"])
    out = capsys.readouterr().out
    assert "Out-of-sample results" in out and "Latest signal" in out
    assert (tmp_path / "results.csv").exists() and (tmp_path / "oos_returns.csv").exists()


def test_csv_loader_accepts_odd_headers(tmp_path):
    p = tmp_path / "x.csv"
    idx = pd.bdate_range("2020-01-01", periods=5)
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "close": [1, 2, 3, 4, 5], "volume": [9] * 5}).to_csv(p, index=False)
    import argparse
    df = tl.load_data(argparse.Namespace(synthetic=False, csv=str(p)))
    assert list(df.columns) == ["Close", "Volume"] and len(df) == 5


def test_too_little_data_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit) as e:
        tl.main(["--synthetic", "--min-train", "5000", "--out", str(tmp_path), "--no-plot"])
    assert "need at least" in str(e.value)
