"""Tests for VCP pattern detection using synthetic series."""

import numpy as np
import pandas as pd

from vcp import vcp_pattern


def _df(close: np.ndarray, volume: np.ndarray | None = None) -> pd.DataFrame:
    n = len(close)
    if volume is None:
        volume = np.full(n, 1000.0)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _make_vcp(uptrend_n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic VCP: uptrend then 3 contractions: 25% -> 12% -> 5%."""
    up = np.linspace(50, 100, uptrend_n)

    seg1 = np.r_[
        np.linspace(100, 105, 5),
        np.linspace(105, 78.75, 10),       # -25%
        np.linspace(78.75, 95, 8),
    ]
    seg2 = np.r_[
        np.linspace(95, 100, 5),
        np.linspace(100, 88, 7),           # -12%
        np.linspace(88, 96, 6),
    ]
    seg3 = np.r_[
        np.linspace(96, 99, 5),
        np.linspace(99, 94.05, 5),         # -5%
        np.linspace(94.05, 97, 4),
    ]
    contractions = np.r_[seg1, seg2, seg3]
    full = np.r_[up, contractions]

    vol = np.r_[
        np.full(uptrend_n, 1000.0),
        np.full(len(seg1), 1500.0),
        np.full(len(seg2), 1000.0),
        np.full(len(seg3), 600.0),
    ]
    return full, vol


def test_clean_vcp_scores_high():
    close, vol = _make_vcp()
    df = _df(close, vol)
    result = vcp_pattern.detect(df, lookback=80)
    assert result.valid, f"expected valid, reason={result.reason}"
    assert result.score >= 0.5
    assert len(result.contractions) >= 2
    mags = result.contractions
    assert all(mags[i] > mags[i + 1] for i in range(len(mags) - 1))


def test_climax_run_extended_flag():
    """A vertical run-up where close is far above SMA50 should set extended flag."""
    n = 250
    close = np.r_[
        np.linspace(50, 60, 200),
        np.linspace(60, 120, 50),
    ]
    df = _df(close)
    result = vcp_pattern.detect(df, lookback=80)
    assert result.extended


def test_no_contractions_invalid():
    n = 250
    close = np.linspace(50, 100, n)
    df = _df(close)
    result = vcp_pattern.detect(df, lookback=80)
    assert not result.valid
    assert result.score == 0.0


def test_increasing_contractions_invalid():
    """T1 < T2 < T3 should fail (contractions must be DEcreasing)."""
    n_up = 200
    up = np.linspace(50, 100, n_up)
    seg1 = np.r_[np.linspace(100, 105, 4), np.linspace(105, 100, 6), np.linspace(100, 105, 5)]
    seg2 = np.r_[np.linspace(105, 110, 4), np.linspace(110, 99, 8), np.linspace(99, 108, 5)]
    seg3 = np.r_[np.linspace(108, 113, 4), np.linspace(113, 90, 10), np.linspace(90, 110, 5)]
    close = np.r_[up, seg1, seg2, seg3]
    df = _df(close)
    result = vcp_pattern.detect(df, lookback=80)
    assert not result.valid
    assert result.reason in {"not_decreasing", "last_contraction_too_big"}
