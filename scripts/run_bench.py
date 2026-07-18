"""Run the INFRADIAN benchmark and write a results JSON.

CP1 usage:  uv run python scripts/run_bench.py --tier C --write-results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from infradian.bench.runner import run_baselines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="C", choices=["C", "B"])
    ap.add_argument("--data", type=Path, default=Path("data/tier_c/cohort.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--write-results", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    res = run_baselines(df, tier=args.tier)

    # Print a compact summary.
    print(f"tier {args.tier}: {res.dataset}")
    for task in ("T2-R", "T2-P"):
        print(f"\n{task} MAE (days), overall stratum:")
        for m in res.metrics:
            if m.task == task and m.stratum == "all":
                print(f"  {m.extra['baseline']:16s} {m.value:6.2f}  (n={m.n})")
        print(f"{task} MAE (days), irregular stratum:")
        for m in res.metrics:
            if m.task == task and m.stratum == "irregular":
                print(f"  {m.extra['baseline']:16s} {m.value:6.2f}  (n={m.n})")

    if args.write_results:
        path = res.write(args.out / f"baselines_tier{args.tier}.json")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
