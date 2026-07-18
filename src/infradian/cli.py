"""`infradian` CLI — thin wrapper over the pipeline commands."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="infradian", description="INFRADIAN benchmark CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("synth", help="generate the Tier C synthetic cohort")
    sp.add_argument("--n", type=int, default=600)
    sp.add_argument("--days", type=int, default=120)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--out", default="data/tier_c")

    sub.add_parser("bench", help="run the benchmark on Tier C")
    sub.add_parser("eval-llm", help="run the explanation-layer eval")

    args = ap.parse_args(argv)

    if args.cmd == "synth":
        from infradian.synth.generator import main as gen
        gen(["--n", str(args.n), "--days", str(args.days), "--seed", str(args.seed), "--out", args.out])
    elif args.cmd == "bench":
        import subprocess
        subprocess.run([sys.executable, "scripts/run_bench.py", "--tier", "C"], check=True)
    elif args.cmd == "eval-llm":
        from infradian.llm.eval import main as ev
        ev()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
