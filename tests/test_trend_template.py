"""Tests for Minervini Trend Template using synthetic price series."""

import numpy as np
import pandas as pd

from vcp import trend_template


def _df_from_close(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0},
        index=idx,
    )


def test_clean_uptrend_passes():
    n = 300
    base = np.linspace(100, 220, n) + np.sin(np.arange(n) / 10) * 2
    df = _df_from_close(base)
    res = trend_template.evaluate(df, rs_rating=85.0)
    d = res.as_dict()
    assert d["above_sma150_200"]
    assert d["sma150_above_sma200"]
    assert d["sma200_trending_up"]
    assert d["sma_stack"]
    assert d["above_sma50"]
    assert d["above_low_30pct"]
    assert d["near_high_25pct"]
    assert d["rs_pass"]
    assert res.passed


def test_downtrend_fails():
    n = 300
    base = np.linspace(220, 100, n)
    df = _df_from_close(base)
    res = trend_template.evaluate(df, rs_rating=20.0)
    assert not res.passed


def test_low_rs_fails_even_if_uptrend():
    n = 300
    base = np.linspace(100, 220, n)
    df = _df_from_close(base)
    res = trend_template.evaluate(df, rs_rating=30.0)
    assert not res.passed
    assert not res.rs_pass


def test_far_from_high_fails():
    """Stock that's pulled back >25% from 52w high should fail."""
    n = 300
    base = np.r_[
        np.linspace(100, 220, 200),
        np.linspace(220, 150, 100),
    ]
    df = _df_from_close(base)
    res = trend_template.evaluate(df, rs_rating=80.0)
    assert not res.near_high_25pct
