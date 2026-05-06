"""Tests for vcp.indicators using synthetic data."""

import numpy as np
import pandas as pd

from vcp.indicators import (
    atr,
    high_52w,
    low_52w,
    rs_rating_panel,
    sma,
    sma_trending_up,
    up_down_volume_ratio,
)


def test_sma_basic():
    s = pd.Series(range(1, 11), dtype=float)
    out = sma(s, 3)
    assert out.iloc[0] != out.iloc[0] or pd.isna(out.iloc[0])
    assert out.iloc[2] == 2.0
    assert out.iloc[-1] == 9.0


def test_atr_increasing_range():
    n = 30
    df = pd.DataFrame(
        {
            "high": np.linspace(10, 12, n),
            "low": np.linspace(9, 10, n),
            "close": np.linspace(9.5, 11, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )
    out = atr(df, n=14)
    assert out.iloc[-1] > 0
    assert pd.isna(out.iloc[5])


def test_high_52w_low_52w_use_243():
    n = 300
    s = pd.Series(np.arange(n, dtype=float), index=pd.date_range("2023-01-01", periods=n, freq="B"))
    h = high_52w(s)
    l = low_52w(s)
    assert h.iloc[-1] == s.iloc[-1]
    assert l.iloc[-1] == s.iloc[-243]


def test_sma_trending_up():
    s = pd.Series(np.linspace(100, 200, 50))
    flag = sma_trending_up(s, days=21)
    assert bool(flag.iloc[-1]) is True

    s_dn = pd.Series(np.linspace(200, 100, 50))
    flag_dn = sma_trending_up(s_dn, days=21)
    assert bool(flag_dn.iloc[-1]) is False


def test_rs_rating_panel_ranks_to_99():
    n = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    closes = pd.DataFrame(
        {
            "A": np.linspace(100, 300, n),  # +200%
            "B": np.linspace(100, 150, n),  # +50%
            "C": np.linspace(100, 110, n),  # +10%
        },
        index=idx,
    )
    rs = rs_rating_panel(closes)
    last = rs.iloc[-1]
    assert last["A"] > last["B"] > last["C"]
    assert last["A"] >= 90  # top performer ~ 99


def test_up_down_volume_ratio():
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    chg = np.tile([1.0, -0.5], n // 2)
    close = pd.Series(100 + np.cumsum(chg), index=idx)
    vol = pd.Series(np.where(chg > 0, 2000.0, 1000.0), index=idx)
    df = pd.DataFrame({"close": close, "volume": vol})
    ratio = up_down_volume_ratio(df, n=50)
    assert ratio.iloc[-1] > 1.5
