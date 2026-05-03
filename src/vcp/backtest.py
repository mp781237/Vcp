"""Simple bar-by-bar backtest for VCP+CANSLIM-lite signals.

Mechanics:
    - Walk forward day by day from `start` to `end`.
    - For efficiency, precompute panel data + RS rating once.
    - Each day: evaluate trend template + VCP + CANSLIM-lite for tickers
      not currently held.
    - Generate signals; rank top N candidates.
    - Enter at next day's open (avoids look-ahead).
    - Position sizing: equal weight across `max_positions` slots.
    - Exit rules (whichever first):
        * stop loss: -7% from entry close
        * profit target: +20% from entry close
        * time stop: held >= 60 trading days
    - Round-trip cost: 0.4% (entry+exit combined, applied at exit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from . import canslim, trend_template, vcp_pattern
from .data import load_many, load_taiex, load_universe
from .indicators import rs_rating_panel
from .screener import _build_close_panel


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    max_positions: int = 5
    stop_loss_pct: float = 0.07
    profit_target_pct: float = 0.20
    max_hold_days: int = 60
    cost_pct: float = 0.004                # round-trip
    min_vcp_score: float = 0.6
    require_all_canslim: bool = False
    rebalance_every_n_days: int = 1        # screen every N bars (1 = daily)


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    capital_allocated: float


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    return_pct: float
    bars_held: int
    exit_reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    config: BacktestConfig = field(repr=False)


def _signals_at(
    as_of: pd.Timestamp,
    data: dict[str, pd.DataFrame],
    rs_panel: pd.DataFrame,
    taiex: pd.DataFrame,
    held: set[str],
    cfg: BacktestConfig,
) -> list[tuple[str, float, float]]:
    """Return [(ticker, rs_rating, vcp_score)] for new entry candidates, ranked."""
    candidates: list[tuple[str, float, float]] = []
    for ticker, df in data.items():
        if ticker in held:
            continue
        df_to_date = df.loc[df.index <= as_of]
        if len(df_to_date) < 200:
            continue
        last_date = df_to_date.index[-1]
        if last_date != as_of:
            continue

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
        candidates.append((ticker, rs_val or 0.0, vp.score))

    candidates.sort(key=lambda x: (-x[1], -x[2]))
    return candidates


def _next_trading_date(
    as_of: pd.Timestamp, data: dict[str, pd.DataFrame], ticker: str
) -> pd.Timestamp | None:
    df = data[ticker]
    future = df.index[df.index > as_of]
    return future[0] if len(future) > 0 else None


def run(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    universe_csv: str = "data/universe.csv",
    cfg: BacktestConfig | None = None,
    refresh: bool = False,
) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    fetch_start = (start_ts - pd.Timedelta(days=460)).date()
    fetch_end = (end_ts + pd.Timedelta(days=10)).date()

    universe = load_universe(universe_csv)
    tickers = universe["yf_ticker"].tolist()
    data = load_many(tickers, fetch_start, fetch_end, refresh=refresh)
    taiex = load_taiex(fetch_start, fetch_end, refresh=refresh)
    if not data:
        raise RuntimeError("no data loaded for backtest")

    closes = _build_close_panel(data)
    rs_panel = rs_rating_panel(closes)

    trading_days: pd.DatetimeIndex = pd.DatetimeIndex(sorted(closes.index.unique()))
    trading_days = trading_days[(trading_days >= start_ts) & (trading_days <= end_ts)]

    cash = cfg.initial_capital
    positions: dict[str, Position] = {}
    trades: list[Trade] = []
    equity_history: dict[pd.Timestamp, float] = {}
    pending_entries: list[tuple[str, pd.Timestamp]] = []  # (ticker, entry_date)

    for i, today in enumerate(trading_days):
        # 1. Process pending entries scheduled for today.
        new_pending: list[tuple[str, pd.Timestamp]] = []
        for ticker, entry_dt in pending_entries:
            if entry_dt != today:
                new_pending.append((ticker, entry_dt))
                continue
            df = data[ticker]
            if today not in df.index or ticker in positions or len(positions) >= cfg.max_positions:
                continue
            entry_price = float(df.loc[today, "open"])
            slot_capital = cfg.initial_capital / cfg.max_positions
            available = min(slot_capital, cash)
            shares = int(available // entry_price)
            if shares <= 0:
                continue
            actual_cost = shares * entry_price
            cash -= actual_cost
            positions[ticker] = Position(
                ticker=ticker,
                entry_date=today,
                entry_price=entry_price,
                shares=shares,
                capital_allocated=actual_cost,
            )
        pending_entries = new_pending

        # 2. Check exit rules for held positions.
        to_close: list[tuple[str, str, float]] = []  # (ticker, reason, exit_price)
        for ticker, pos in positions.items():
            df = data[ticker]
            if today not in df.index:
                continue
            row = df.loc[today]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            stop_price = pos.entry_price * (1 - cfg.stop_loss_pct)
            target_price = pos.entry_price * (1 + cfg.profit_target_pct)
            bars_held = trading_days.get_loc(today) - trading_days.get_loc(pos.entry_date)

            if low <= stop_price:
                to_close.append((ticker, "stop_loss", stop_price))
            elif high >= target_price:
                to_close.append((ticker, "profit_target", target_price))
            elif bars_held >= cfg.max_hold_days:
                to_close.append((ticker, "time_stop", close))

        for ticker, reason, exit_price in to_close:
            pos = positions.pop(ticker)
            gross = pos.shares * exit_price
            cost = (pos.capital_allocated + gross) * cfg.cost_pct / 2
            cash += gross - cost
            pnl = gross - pos.capital_allocated - cost
            ret = pnl / pos.capital_allocated
            bars_held = trading_days.get_loc(today) - trading_days.get_loc(pos.entry_date)
            trades.append(
                Trade(
                    ticker=ticker,
                    entry_date=pos.entry_date,
                    exit_date=today,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    shares=pos.shares,
                    pnl=pnl,
                    return_pct=ret,
                    bars_held=bars_held,
                    exit_reason=reason,
                )
            )

        # 3. Generate new signals (rebalance schedule).
        if i % cfg.rebalance_every_n_days == 0 and len(positions) < cfg.max_positions:
            held = set(positions.keys()) | {t for t, _ in pending_entries}
            slots_left = cfg.max_positions - len(positions) - len(pending_entries)
            if slots_left > 0:
                cands = _signals_at(today, data, rs_panel, taiex, held, cfg)
                for ticker, _, _ in cands[:slots_left]:
                    next_date = _next_trading_date(today, data, ticker)
                    if next_date is not None and next_date <= end_ts:
                        pending_entries.append((ticker, next_date))

        # 4. Mark to market.
        equity = cash
        for ticker, pos in positions.items():
            df = data[ticker]
            if today in df.index:
                equity += pos.shares * float(df.loc[today, "close"])
            else:
                equity += pos.capital_allocated
        equity_history[today] = equity

    # Force-close any remaining positions at end.
    if positions and len(trading_days) > 0:
        last_day = trading_days[-1]
        for ticker, pos in list(positions.items()):
            df = data[ticker]
            if last_day in df.index:
                exit_price = float(df.loc[last_day, "close"])
            else:
                exit_price = pos.entry_price
            gross = pos.shares * exit_price
            cost = (pos.capital_allocated + gross) * cfg.cost_pct / 2
            cash += gross - cost
            pnl = gross - pos.capital_allocated - cost
            ret = pnl / pos.capital_allocated
            bars_held = trading_days.get_loc(last_day) - trading_days.get_loc(pos.entry_date)
            trades.append(
                Trade(
                    ticker=ticker,
                    entry_date=pos.entry_date,
                    exit_date=last_day,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    shares=pos.shares,
                    pnl=pnl,
                    return_pct=ret,
                    bars_held=bars_held,
                    exit_reason="end_of_test",
                )
            )

    equity_curve = pd.Series(equity_history, name="equity").sort_index()
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    return BacktestResult(equity_curve=equity_curve, trades=trades_df, config=cfg)
