"""Produce every published INFRADIAN result: full-cohort Tier C, Tier B (real), and the sim-to-real
transfer, plus the committable synthetic-trained reference bundle. Writes results/*.json.

Usage:
  uv run python scripts/run_full_bench.py --mcphases <path>   # full run (needs DUA data)
  uv run python scripts/run_full_bench.py                     # Tier C only (no data access)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from infradian.bench.model_runner import build_results, cross_validate_model, transfer_evaluate
from infradian.bench.runner import run_baselines
from infradian.models.reference import save_bundle, train_bundle


def _print(res, title):
    print(f"\n=== {title} ===")
    for m in res.metrics:
        extra = ""
        if m.metric == "SoC":
            extra = f" [D_i={m.extra.get('median_D_i', 0):+.2f}d p={m.extra.get('wilcoxon_p', 1):.3f}]"
        if m.task == "T2-A":
            extra = f" [prev={m.extra.get('prevalence', 0):.2f} lift=+{m.extra.get('lift_over_calendar', 0):.2f}]"
        ci = f" CI[{m.ci_lo:.2f},{m.ci_hi:.2f}]" if m.ci_lo is not None else ""
        star = " *PRIMARY*" if m.is_primary else ""
        print(f"  {m.task:8s} {m.stratum:9s} {m.metric:24s} {m.value:+.3f}{ci} n={m.n}{extra}{star}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcphases", type=Path, default=None)
    ap.add_argument("--tier-c", type=Path, default=Path("data/tier_c/cohort.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    df_c = pd.read_parquet(args.tier_c)

    # ---- Tier C: baselines + model (the reproducible-by-anyone headline) ----
    run_baselines(df_c, tier="C").write(args.out / "baselines_tierC.json")
    pred_c = cross_validate_model(df_c, arm="wearable_menses")
    res_c = build_results(pred_c, df_c, tier="C", run_name="model-tierC", use_metadata=True)
    res_c.write(args.out / "model_tierC.json")
    _print(res_c, "Tier C (synthetic, reproducible by anyone)")

    # ---- Reference bundle: trained on SYNTHETIC only (the publishable checkpoint) ----
    bundle = train_bundle(df_c, arm="wearable_menses")
    save_bundle(bundle, args.out / "models" / "infradian-ref-s")
    print("\nsaved synthetic-trained reference bundle -> results/models/infradian-ref-s")

    # ---- Tier B: real data (only with DUA access) ----
    if args.mcphases and args.mcphases.exists():
        from infradian.data.mcphases import coverage_report, load_canonical

        df_b = load_canonical(args.mcphases)
        cov = coverage_report(df_b)
        (args.out / "mcphases_aggregates.json").write_text(pd.Series(cov).to_json(indent=2))

        run_baselines(df_b, tier="B", run_name="baselines-real").write(args.out / "baselines_tierB.json")
        pred_b = cross_validate_model(df_b, arm="wearable_menses")
        res_b = build_results(pred_b, df_b, tier="B", run_name="model-tierB", use_metadata=False)
        res_b.write(args.out / "model_tierB.json")
        _print(res_b, "Tier B (real mcPHASES)")

        # ---- Sim-to-real transfer: train on synthetic, evaluate on real ----
        pred_t = transfer_evaluate(df_c, df_b, arm="wearable_menses")
        res_t = build_results(pred_t, df_b, tier="B", run_name="transfer-sim2real", use_metadata=False)
        res_t.notes = "Trained on Tier C synthetic only; evaluated on real mcPHASES. Sim-to-real transfer."
        res_t.write(args.out / "transfer_sim2real.json")
        _print(res_t, "Sim-to-real transfer (synthetic-trained -> real)")


if __name__ == "__main__":
    main()
