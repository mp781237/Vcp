"""VCP (Volatility Contraction Pattern) detection.

Algorithm (Minervini-style, formalized):

1. In the last `lookback` bars, find swing highs and swing lows on close.
2. Pair them into contraction segments: each segment is a peak followed by a trough.
3. A valid VCP requires:
   - >= 2 contractions
   - magnitudes are monotonically (weakly) decreasing: T1 > T2 > T3 ...
   - the last contraction is small (<= max_last_pct, default 12%)
   - volume on the last contraction is lower than the first (volume dry-up)
4. Score in [0, 1]:
   - base 0.5 if >= 2 strictly decreasing contractions
   - +0.2 if >= 3 contractions
   - +0.2 if last contraction <= 8%
   - +0.1 if volume drops >= 30% from first to last contraction
   - 0 if invalid
5. extended_flag: close > sma50 * 1.25 (close is "extended" from base — risky entry).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .indicators import sma


@dataclass
class VcpResult:
    score: float
    contractions: list[float]      # each in [0, 1] e.g. [0.25, 0.15, 0.07]
    extended: bool
    valid: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "vcp_score": self.score,
            "contractions": [round(c, 4) for c in self.contractions],
            "extended": self.extended,
            "valid": self.valid,
            "reason": self.reason,
        }


def _find_swings(close: np.ndarray, distance: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (peak_indices, trough_indices) using scipy.find_peaks."""
    peaks, _ = find_peaks(close, distance=distance)
    troughs, _ = find_peaks(-close, distance=distance)
    return peaks, troughs


def _build_contractions(
    close: np.ndarray, peaks: np.ndarray, troughs: np.ndarray
) -> list[tuple[int, int, float]]:
    """Pair each peak with the next trough; return [(peak_idx, trough_idx, magnitude)]."""
    contractions: list[tuple[int, int, float]] = []
    for p_idx in peaks:
        next_troughs = troughs[troughs > p_idx]
        if len(next_troughs) == 0:
            continue
        t_idx = int(next_troughs[0])
        peak_price = float(close[p_idx])
        trough_price = float(close[t_idx])
        if peak_price <= 0:
            continue
        mag = (peak_price - trough_price) / peak_price
        if mag > 0:
            contractions.append((int(p_idx), int(t_idx), mag))
    return contractions


def detect(
    df: pd.DataFrame,
    lookback: int = 90,
    swing_distance: int = 5,
    max_last_pct: float = 0.12,
    extended_threshold: float = 0.25,
) -> VcpResult:
    """Detect VCP at the last bar of df.

    df : OHLCV with at least lookback + 50 bars of history (50 for SMA50).
    lookback : bars to scan for contractions (default 90 ~= 4 months).
    """
    if len(df) < lookback + 50:
        return VcpResult(0.0, [], False, False, "insufficient_history")

    window = df.iloc[-lookback:]
    close_arr = window["close"].to_numpy(dtype=float)
    vol_arr = window["volume"].to_numpy(dtype=float)

    peaks, troughs = _find_swings(close_arr, distance=swing_distance)
    contractions = _build_contractions(close_arr, peaks, troughs)

    s50 = sma(df["close"], 50)
    last_close = float(df["close"].iloc[-1])
    last_sma50 = float(s50.iloc[-1]) if pd.notna(s50.iloc[-1]) else float("nan")
    extended = (
        not np.isnan(last_sma50)
        and last_sma50 > 0
        and last_close > last_sma50 * (1.0 + extended_threshold)
    )

    if len(contractions) < 2:
        return VcpResult(0.0, [c[2] for c in contractions], extended, False, "fewer_than_2_contractions")

    mags = [c[2] for c in contractions]
    strictly_decreasing = all(mags[i] > mags[i + 1] for i in range(len(mags) - 1))
    last_mag = mags[-1]
    last_too_big = last_mag > max_last_pct

    first_peak_idx, first_trough_idx, _ = contractions[0]
    last_peak_idx, last_trough_idx, _ = contractions[-1]
    first_vol = float(np.mean(vol_arr[first_peak_idx : first_trough_idx + 1]))
    last_vol = float(np.mean(vol_arr[last_peak_idx : last_trough_idx + 1]))
    vol_dryup = first_vol > 0 and (last_vol / first_vol) < 1.0

    if not strictly_decreasing:
        return VcpResult(0.0, mags, extended, False, "not_decreasing")
    if last_too_big:
        return VcpResult(0.0, mags, extended, False, "last_contraction_too_big")
    if not vol_dryup:
        return VcpResult(0.0, mags, extended, False, "no_volume_dryup")

    score = 0.5
    if len(mags) >= 3:
        score += 0.2
    if last_mag <= 0.08:
        score += 0.2
    if first_vol > 0 and (last_vol / first_vol) <= 0.7:
        score += 0.1

    return VcpResult(min(score, 1.0), mags, extended, True, "ok")
