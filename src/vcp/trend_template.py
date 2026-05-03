"""Minervini Trend Template — 8 criteria for stage-2 uptrend.

Source: 'Trade Like a Stock Market Wizard' (Mark Minervini), Ch. 4.

Adapted for Taiwan: 52-week window uses 243 bars (TRADING_DAYS_PER_YEAR).
RS rating is supplied externally (cross-sectional rank, 0-99).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import (
    high_52w,
    low_52w,
    sma,
    sma_trending_up,
)


@dataclass
class TrendTemplateResult:
    above_sma150_200: bool
    sma150_above_sma200: bool
    sma200_trending_up: bool
    sma_stack: bool          # SMA50 > SMA150 > SMA200
    above_sma50: bool
    above_low_30pct: bool    # close >= 52w_low * 1.30
    near_high_25pct: bool    # close >= 52w_high * 0.75
    rs_pass: bool            # RS rating >= 70

    @property
    def passed(self) -> bool:
        return all(
            [
                self.above_sma150_200,
                self.sma150_above_sma200,
                self.sma200_trending_up,
                self.sma_stack,
                self.above_sma50,
                self.above_low_30pct,
                self.near_high_25pct,
                self.rs_pass,
            ]
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "above_sma150_200": self.above_sma150_200,
            "sma150_above_sma200": self.sma150_above_sma200,
            "sma200_trending_up": self.sma200_trending_up,
            "sma_stack": self.sma_stack,
            "above_sma50": self.above_sma50,
            "above_low_30pct": self.above_low_30pct,
            "near_high_25pct": self.near_high_25pct,
            "rs_pass": self.rs_pass,
            "passed": self.passed,
        }


def evaluate(df: pd.DataFrame, rs_rating: float | None) -> TrendTemplateResult:
    """Evaluate trend template at the last bar of df.

    df : OHLCV with at least 200+ bars of history (preferably 250+).
    rs_rating : cross-sectional RS rating (0-99) for this ticker on this date.
                If None, rs_pass defaults to False.
    """
    close = df["close"]
    s50 = sma(close, 50)
    s150 = sma(close, 150)
    s200 = sma(close, 200)
    s200_up = sma_trending_up(s200, days=21)
    h52 = high_52w(close)
    l52 = low_52w(close)

    last = df.index[-1]
    c = float(close.loc[last])
    m50 = float(s50.loc[last]) if pd.notna(s50.loc[last]) else float("nan")
    m150 = float(s150.loc[last]) if pd.notna(s150.loc[last]) else float("nan")
    m200 = float(s200.loc[last]) if pd.notna(s200.loc[last]) else float("nan")
    h = float(h52.loc[last])
    lo = float(l52.loc[last])

    def _bool(x: bool | float) -> bool:
        return bool(x) and not pd.isna(x)

    return TrendTemplateResult(
        above_sma150_200=_bool(c > m150 and c > m200),
        sma150_above_sma200=_bool(m150 > m200),
        sma200_trending_up=_bool(s200_up.loc[last]),
        sma_stack=_bool(m50 > m150 > m200),
        above_sma50=_bool(c > m50),
        above_low_30pct=_bool(c >= lo * 1.30),
        near_high_25pct=_bool(c >= h * 0.75),
        rs_pass=_bool((rs_rating is not None) and (rs_rating >= 70)),
    )
