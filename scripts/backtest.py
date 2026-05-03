"""CLI: run backtest over a date range.

Usage:
    python -m scripts.backtest --start 2020-01-01 --end 2024-12-31
    python -m scripts.backtest --start 2020-01-01 --end 2024-12-31 \\
        --trades-out out/trades.csv --equity-out out/equity.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vcp.backtest import BacktestConfig, run
from vcp.metrics import format_summary, summary


def main() -> int:
    p = argparse.ArgumentParser(description="VCP+CANSLIM backtest")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--universe", default="data/universe.csv")
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--stop-loss", type=float, default=0.07)
    p.add_argument("--profit-target", type=float, default=0.20)
    p.add_argument("--max-hold", type=int, default=60)
    p.add_argument("--cost", type=float, default=0.004)
    p.add_argument("--min-vcp", type=float, default=0.6)
    p.add_argument("--all-canslim", action="store_true")
    p.add_argument("--rebalance-every", type=int, default=1)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--trades-out", default=None)
    p.add_argument("--equity-out", default=None)
    args = p.parse_args()

    cfg = BacktestConfig(
        initial_capital=args.capital,
        max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss,
        profit_target_pct=args.profit_target,
        max_hold_days=args.max_hold,
        cost_pct=args.cost,
        min_vcp_score=args.min_vcp,
        require_all_canslim=args.all_canslim,
        rebalance_every_n_days=args.rebalance_every,
    )

    print(f"Running backtest {args.start} -> {args.end} ...", file=sys.stderr)
    result = run(args.start, args.end, universe_csv=args.universe, cfg=cfg, refresh=args.refresh)

    if args.trades_out:
        Path(args.trades_out).parent.mkdir(parents=True, exist_ok=True)
        result.trades.to_csv(args.trades_out, index=False)
        print(f"Wrote {len(result.trades)} trades to {args.trades_out}", file=sys.stderr)

    if args.equity_out:
        Path(args.equity_out).parent.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(args.equity_out, header=True)
        print(f"Wrote equity curve ({len(result.equity_curve)} bars) to {args.equity_out}", file=sys.stderr)

    s = summary(result.equity_curve, result.trades)
    print()
    print("Backtest summary:")
    print(format_summary(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
