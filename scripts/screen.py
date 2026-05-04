"""CLI: run VCP+CANSLIM screener for a given date.

Usage:
    python -m scripts.screen --date 2024-12-31
    python -m scripts.screen --date 2024-12-31 --output out/screen.csv
    python -m scripts.screen --date 2024-04-30 --explain 6531.TW
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vcp.screener import ScreenerConfig, explain, screen_at, screen_recent_at


def main() -> int:
    p = argparse.ArgumentParser(description="VCP+CANSLIM screener")
    p.add_argument("--date", required=True, help="As-of date YYYY-MM-DD")
    p.add_argument("--universe", default="data/universe.csv")
    p.add_argument("--output", default=None, help="Optional CSV path")
    p.add_argument("--min-vcp", type=float, default=0.6)
    p.add_argument("--all-canslim", action="store_true", help="Require all of L/S/N/M")
    p.add_argument("--refresh", action="store_true", help="Force re-download data")
    p.add_argument("--explain", default=None, help="Show diagnostic for a single ticker")
    p.add_argument(
        "--recent",
        type=int,
        default=0,
        metavar="N",
        help="Look back N trading bars; list any ticker that triggered within that window",
    )
    args = p.parse_args()

    if args.explain:
        diag = explain(args.explain, args.date, universe_csv=args.universe, refresh=args.refresh)
        print(json.dumps(diag, indent=2, default=str, ensure_ascii=False))
        return 0

    cfg = ScreenerConfig(min_vcp_score=args.min_vcp, require_all_canslim=args.all_canslim)
    if args.recent > 0:
        df = screen_recent_at(
            args.date,
            lookback_days=args.recent,
            universe_csv=args.universe,
            cfg=cfg,
            refresh=args.refresh,
        )
    else:
        df = screen_at(args.date, universe_csv=args.universe, cfg=cfg, refresh=args.refresh)

    if df.empty:
        print(f"No matches for {args.date}.", file=sys.stderr)
        return 0

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} rows to {args.output}", file=sys.stderr)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
