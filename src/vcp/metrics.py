"""Performance metrics for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import TRADING_DAYS_PER_YEAR


def cagr(equity: pd.Series) -> float:
    if equity.empty or len(equity) < 2:
        return 0.0
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val <= 0:
        return 0.0
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    return (end_val / start_val) ** (1 / years) - 1


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return float(dd.min())


def sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    if equity.empty or len(equity) < 2:
        return 0.0
    rets = equity.pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    excess = rets - rf / TRADING_DAYS_PER_YEAR
    return float((excess.mean() / rets.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "avg_hold_bars": 0.0,
        }

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    avg_win = float(wins["return_pct"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["return_pct"].mean()) if not losses.empty else 0.0
    win_rate = len(wins) / len(trades)
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["pnl"].sum()) if not losses.empty else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return {
        "n_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": pf,
        "avg_hold_bars": float(trades["bars_held"].mean()),
    }


def summary(equity: pd.Series, trades: pd.DataFrame) -> dict:
    out = {
        "cagr": cagr(equity),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe(equity),
        "final_equity": float(equity.iloc[-1]) if not equity.empty else 0.0,
    }
    out.update(trade_stats(trades))
    return out


def format_summary(s: dict) -> str:
    lines = [
        f"  CAGR            : {s['cagr']:>8.2%}",
        f"  Max Drawdown    : {s['max_drawdown']:>8.2%}",
        f"  Sharpe          : {s['sharpe']:>8.2f}",
        f"  Final Equity    : {s['final_equity']:>12,.0f}",
        f"  N Trades        : {s['n_trades']:>8d}",
        f"  Win Rate        : {s['win_rate']:>8.2%}",
        f"  Avg Win         : {s['avg_win']:>8.2%}",
        f"  Avg Loss        : {s['avg_loss']:>8.2%}",
        f"  Expectancy      : {s['expectancy']:>8.2%}",
        f"  Profit Factor   : {s['profit_factor']:>8.2f}",
        f"  Avg Hold (bars) : {s['avg_hold_bars']:>8.1f}",
    ]
    return "\n".join(lines)
