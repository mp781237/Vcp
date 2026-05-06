"""Tests for screener helpers."""

import numpy as np
import pandas as pd

from vcp.screener import _is_extended_now


def _df(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def test_is_extended_now_climax_run():
    """A vertical run-up where close is far above SMA50 should be extended."""
    n = 200
    close = np.r_[
        np.linspace(50, 60, 150),
        np.linspace(60, 120, 50),
    ]
    assert _is_extended_now(_df(close)) is True


def test_is_extended_now_normal_uptrend():
    """A gentle uptrend close to SMA50 should NOT be extended."""
    n = 200
    close = np.linspace(50, 100, n)
    assert _is_extended_now(_df(close)) is False


def test_is_extended_now_handles_short_series():
    """If we don't have 50 bars yet, can't compute SMA50 — return False (safe default)."""
    close = np.linspace(50, 60, 30)
    assert _is_extended_now(_df(close)) is False
