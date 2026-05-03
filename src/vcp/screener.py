"""Screener: compose Trend Template + CANSLIM-lite + VCP pattern."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import canslim, trend_template, vcp_pattern
from .data import load_many, load_taiex, load_universe
from .indicators import rs_rating_panel


@dataclass
class ScreenerConfig:
    min_vcp_score: float = 0.6
    require_all_canslim: bool = False  # if False, only L+M required
    history_days: int = 400            # how much history to fetch (need 243+ for SMA200/RS)


def _build_close_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack close series across tickers into a single DataFrame (date x ticker)."""
    if not data:
        return pd.DataFrame()
    closes = pd.concat({t: d["close"] for t, d in data.items()}, axis=1)
    closes.columns = list(data.keys())
    return closes


def screen_at(
    as_of: str | pd.Timestamp,
    universe_csv: str = "data/universe.csv",
    cfg: ScreenerConfig | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Run screener for a single date.

    Returns DataFrame ranked by RS desc, vcp_score desc.
    """
    cfg = cfg or ScreenerConfig()
    as_of_ts = pd.Timestamp(as_of)
    start = (as_of_ts - pd.Timedelta(days=cfg.history_days + 60)).date()
    end = (as_of_ts + pd.Timedelta(days=1)).date()

    universe = load_universe(universe_csv)
    sym_to_name = dict(zip(universe["yf_ticker"], universe["name"]))
    tickers = universe["yf_ticker"].tolist()

    data = load_many(tickers, start, end, refresh=refresh)
    if not data:
        return pd.DataFrame()

    taiex = load_taiex(start, end, refresh=refresh)

    closes = _build_close_panel(data)
    rs_panel = rs_rating_panel(closes)

    rows: list[dict] = []
    for ticker, df in data.items():
        df_to_date = df.loc[df.index <= as_of_ts]
        if len(df_to_date) < 200:
            continue
        last_date = df_to_date.index[-1]

        rs_val = None
        if last_date in rs_panel.index and ticker in rs_panel.columns:
            v = rs_panel.loc[last_date, ticker]
            if pd.notna(v):
                rs_val = float(v)

        tt = trend_template.evaluate(df_to_date, rs_val)
        if not tt.passed:
            continue

        vp = vcp_pattern.detect(df_to_date)
        if not vp.valid or vp.score < cfg.min_vcp_score or vp.extended:
            continue

        cs = canslim.evaluate(df_to_date, rs_val, taiex)
        canslim_pass = cs.all_passed if cfg.require_all_canslim else cs.passed_core
        if not canslim_pass:
            continue

        rows.append(
            {
                "symbol": ticker.split(".")[0],
                "name": sym_to_name.get(ticker, ""),
                "ticker": ticker,
                "date": last_date.strftime("%Y-%m-%d"),
                "close": round(float(df_to_date["close"].iloc[-1]), 2),
                "rs": round(rs_val, 1) if rs_val is not None else None,
                "vcp_score": round(vp.score, 2),
                "contractions": ", ".join(f"{c:.1%}" for c in vp.contractions),
                "extended": vp.extended,
                "L": cs.L_leader,
                "S": cs.S_supply_demand,
                "N": cs.N_new_high,
                "M": cs.M_market_uptrend,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["rs", "vcp_score"], ascending=[False, False]).reset_index(drop=True)


def explain(
    ticker: str,
    as_of: str | pd.Timestamp,
    universe_csv: str = "data/universe.csv",
    refresh: bool = False,
) -> dict:
    """Detailed per-ticker diagnostic at a given date."""
    as_of_ts = pd.Timestamp(as_of)
    start = (as_of_ts - pd.Timedelta(days=460)).date()
    end = (as_of_ts + pd.Timedelta(days=1)).date()

    universe = load_universe(universe_csv)
    if ticker not in universe["yf_ticker"].values:
        if ticker.isdigit():
            row = universe[universe["symbol"] == ticker]
            if not row.empty:
                ticker = row.iloc[0]["yf_ticker"]

    data = load_many(universe["yf_ticker"].tolist(), start, end, refresh=refresh)
    taiex = load_taiex(start, end, refresh=refresh)

    if ticker not in data:
        return {"error": f"no data for {ticker}"}

    df = data[ticker].loc[data[ticker].index <= as_of_ts]
    closes = _build_close_panel(data)
    rs_panel = rs_rating_panel(closes)
    last = df.index[-1]
    rs_val = None
    if last in rs_panel.index and ticker in rs_panel.columns:
        v = rs_panel.loc[last, ticker]
        if pd.notna(v):
            rs_val = float(v)

    tt = trend_template.evaluate(df, rs_val)
    vp = vcp_pattern.detect(df)
    cs = canslim.evaluate(df, rs_val, taiex)

    return {
        "ticker": ticker,
        "date": last.strftime("%Y-%m-%d"),
        "close": round(float(df["close"].iloc[-1]), 2),
        "rs_rating": rs_val,
        "trend_template": tt.as_dict(),
        "vcp": vp.as_dict(),
        "canslim_lite": cs.as_dict(),
    }
