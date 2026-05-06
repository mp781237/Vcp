"""CANSLIM-lite: price/volume/market-direction subset of O'Neil's CANSLIM.

This implements only the components that don't need fundamentals data:
    L : Leader        — RS rating >= 80
    S : Supply/demand — up-day volume > down-day volume in last 50 bars
    N : New high      — close within 8% of 52-week high
    M : Market dir    — TAIEX above its 200-day SMA

Skipped (require fundamentals / institutional data):
    C : Current quarterly EPS growth
    A : Annual EPS growth
    I : Institutional sponsorship
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import high_52w, sma, up_down_volume_ratio


@dataclass
class CanslimLiteResult:
    L_leader: bool
    S_supply_demand: bool
    N_new_high: bool
    M_market_uptrend: bool

    @property
    def passed_core(self) -> bool:
        """L + M are the must-pass core (leader in an uptrend market)."""
        return self.L_leader and self.M_market_uptrend

    @property
    def all_passed(self) -> bool:
        return self.L_leader and self.S_supply_demand and self.N_new_high and self.M_market_uptrend

    def as_dict(self) -> dict[str, bool]:
        return {
            "L_leader": self.L_leader,
            "S_supply_demand": self.S_supply_demand,
            "N_new_high": self.N_new_high,
            "M_market_uptrend": self.M_market_uptrend,
            "canslim_core": self.passed_core,
            "canslim_all": self.all_passed,
        }


def evaluate(
    df: pd.DataFrame,
    rs_rating: float | None,
    taiex: pd.DataFrame,
    near_high_pct: float = 0.08,
) -> CanslimLiteResult:
    """Evaluate CANSLIM-lite at the last bar of df.

    df : ticker OHLCV
    rs_rating : cross-sectional RS rating (0-99) for this ticker on this date
    taiex : TAIEX OHLCV (index ^TWII), used for M
    near_high_pct : how close to 52w high to count as 'new high' (default 8%)
    """
    last = df.index[-1]
    close = df["close"]
    c = float(close.loc[last])
    h = float(high_52w(close).loc[last])

    udvr = up_down_volume_ratio(df, n=50)
    s_pass = bool(pd.notna(udvr.loc[last]) and udvr.loc[last] > 1.0)

    n_pass = bool(h > 0 and c >= h * (1.0 - near_high_pct))

    l_pass = rs_rating is not None and rs_rating >= 80

    if not taiex.empty and last in taiex.index:
        tx_close = taiex["close"]
        tx_sma200 = sma(tx_close, 200)
        m_pass = bool(
            pd.notna(tx_sma200.loc[last]) and tx_close.loc[last] > tx_sma200.loc[last]
        )
    else:
        tx_close = taiex["close"]
        tx_sma200 = sma(tx_close, 200)
        idx = tx_close.index[tx_close.index <= last]
        if len(idx) > 0:
            d = idx[-1]
            m_pass = bool(pd.notna(tx_sma200.loc[d]) and tx_close.loc[d] > tx_sma200.loc[d])
        else:
            m_pass = False

    return CanslimLiteResult(
        L_leader=bool(l_pass),
        S_supply_demand=s_pass,
        N_new_high=n_pass,
        M_market_uptrend=m_pass,
    )
