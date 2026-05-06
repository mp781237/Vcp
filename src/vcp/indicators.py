"""Technical indicators and cross-sectional rankings.

All "year" calculations use TRADING_DAYS_PER_YEAR = 243 (Taiwan trading year).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import TRADING_DAYS_PER_YEAR


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range. Expects columns: high, low, close."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def rolling_max(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).max()


def rolling_min(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).min()


def high_52w(close: pd.Series) -> pd.Series:
    return rolling_max(close, TRADING_DAYS_PER_YEAR)


def low_52w(close: pd.Series) -> pd.Series:
    return rolling_min(close, TRADING_DAYS_PER_YEAR)


def returns_lookback(close: pd.Series, lookback: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """Total return over `lookback` bars: close / close[-lookback] - 1."""
    return close / close.shift(lookback) - 1.0


def rs_rating_panel(closes: pd.DataFrame, lookback: int = TRADING_DAYS_PER_YEAR) -> pd.DataFrame:
    """Cross-sectional RS rating (0-99) per date.

    closes : DataFrame indexed by date, columns are tickers.
    For each row, compute lookback-period return and rank to a percentile in [0, 99].
    Tickers with missing data are excluded from that day's ranking.
    """
    rets = closes / closes.shift(lookback) - 1.0
    ranks = rets.rank(axis=1, pct=True, method="average")
    return (ranks * 99).round().astype("Float64")


def sma_trending_up(s: pd.Series, days: int = 21) -> pd.Series:
    """Boolean series: True where s is above its value `days` bars ago."""
    return s > s.shift(days)


def up_down_volume_ratio(df: pd.DataFrame, n: int = 50) -> pd.Series:
    """Sum of volume on up days / sum of volume on down days, over last n bars.

    Used for CANSLIM 'S' (supply/demand). Ratio > 1 means accumulation.
    """
    close = df["close"]
    vol = df["volume"]
    chg = close.diff()
    up_vol = vol.where(chg > 0, 0.0)
    dn_vol = vol.where(chg < 0, 0.0)
    up_sum = up_vol.rolling(n, min_periods=n).sum()
    dn_sum = dn_vol.rolling(n, min_periods=n).sum()
    return up_sum / dn_sum.replace(0, np.nan)


def percent_from(close: pd.Series, ref: pd.Series) -> pd.Series:
    """(close - ref) / ref."""
    return (close - ref) / ref
