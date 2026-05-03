"""yfinance downloader with parquet cache.

Cache layout:
    data/cache/<ticker>.parquet   # full history per yfinance ticker (e.g. 2330.TW)

Re-download triggers:
    - cache file missing
    - cache file's last date < requested end date
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data/cache")


def to_yf_ticker(symbol: str, market: str) -> str:
    """Convert TW symbol + market to yfinance ticker."""
    suffix = ".TW" if market == "TWSE" else ".TWO"
    return f"{symbol}{suffix}"


def load_universe(path: str | Path = "data/universe.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"symbol": str})
    df["yf_ticker"] = df.apply(lambda r: to_yf_ticker(r["symbol"], r["market"]), axis=1)
    return df


def _cache_path(yf_ticker: str) -> Path:
    return CACHE_DIR / f"{yf_ticker}.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize yfinance output: lowercase columns, drop rows with NaN close/volume."""
    if df.empty:
        return df
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    expected = {"open", "high", "low", "close", "volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"missing columns from yfinance: {missing}")
    df = df.dropna(subset=["close", "volume"])
    df = df[df["volume"] > 0]
    df.index.name = "date"
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def download(yf_ticker: str, start: str | date, end: str | date, retries: int = 3) -> pd.DataFrame:
    """Download via yfinance with retries on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(
                yf_ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return _normalize(df)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(1 + attempt * 2)
    raise RuntimeError(f"download failed for {yf_ticker}: {last_exc}")


def load(yf_ticker: str, start: str | date, end: str | date, refresh: bool = False) -> pd.DataFrame:
    """Load OHLCV from cache, downloading/extending if needed.

    Returns a DataFrame indexed by date with columns: open, high, low, close, volume, adj_close.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(yf_ticker)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    cached: pd.DataFrame | None = None
    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        cached.index = pd.to_datetime(cached.index)
        if not cached.empty and cached.index.max() >= end_ts - pd.Timedelta(days=1):
            return cached.loc[start_ts:end_ts]

    fetch_start = start_ts - pd.Timedelta(days=400)
    fetch_end = end_ts + pd.Timedelta(days=1)
    fresh = download(yf_ticker, fetch_start.date(), fetch_end.date())

    if cached is not None and not cached.empty:
        merged = pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fresh

    if not merged.empty:
        merged.to_parquet(path)
    return merged.loc[start_ts:end_ts]


def load_many(
    yf_tickers: list[str],
    start: str | date,
    end: str | date,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Bulk load. Skips tickers that fail (logs to stderr)."""
    out: dict[str, pd.DataFrame] = {}
    for t in yf_tickers:
        try:
            df = load(t, start, end, refresh=refresh)
            if not df.empty:
                out[t] = df
        except Exception as e:  # noqa: BLE001
            print(f"[load_many] {t} failed: {e}")
    return out


def load_taiex(start: str | date, end: str | date, refresh: bool = False) -> pd.DataFrame:
    """Load TAIEX index for market filter (M of CANSLIM)."""
    return load("^TWII", start, end, refresh=refresh)
