"""Export static JSON for the frontend. The demo path is 100% static — no live inference — so it
cannot fail during a demo. Every published trajectory is a SYNTHETIC participant (Tier C, CC-BY);
real mcPHASES trajectories are never plotted, only its aggregate metrics appear.

Writes web/public/data/{manifest,participants/*,skill,leaderboard,explanations}.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from infradian.bench.model_runner import transfer_evaluate
from infradian.bench.tasks import cycles_to_frame, extract_cycles
from infradian.data import canonical as C

WEB = Path("web/public/data")
RESULTS = Path("results")


def _round(x, n=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), n)


def _series_band(mean: np.ndarray, resid_std: float):
    """90% band from the model's empirical residual std (honest prediction error, not decoration)."""
    lo = mean - 1.64 * resid_std
    hi = mean + 1.64 * resid_std
    return (
        [_round(v) for v in mean],
        [_round(max(0, v)) for v in lo],
        [_round(v) for v in hi],
    )


def export_participant(pid, g, pred, resid_std, cycles) -> dict:
    idx = g.index.to_numpy()
    days = g[C.KEY_DAY].astype(int).tolist()
    out = {
        "pid": pid,
        "regularity": cycles.iloc[0]["regularity"] if len(cycles) else "regular",
        "days": days,
        "truth": {},
        "model": {},
        "cycles": [],
        "events": [],
    }
    for h in ("e3g", "pdg", "lh"):
        truth = g[h].to_numpy()
        out["truth"][h] = [_round(v) for v in truth]
    for h, arr in (("pdg", pred.pdg), ("e3g", pred.e3g)):
        mean = np.nan_to_num(arr[idx])
        m, lo, hi = _series_band(mean, resid_std[h])
        out["model"][h] = {"mean": m, "lo": lo, "hi": hi}
    # LH: show the model's ovulation-probability-scaled surge estimate as a light overlay
    ovp = pred.phase_prob[idx, C.PHASES.index("ovulation")]
    lh_scale = np.nanmax(g["lh"].to_numpy()) if g["lh"].notna().any() else 1.0
    out["model"]["lh"] = {"mean": [_round(v * lh_scale) for v in np.nan_to_num(ovp)], "lo": [], "hi": []}

    for _, c in cycles.iterrows():
        out["cycles"].append(
            {"index": int(len(out["cycles"]) + 1), "startDay": int(c["cycle_start"]),
             "endDay": int(c["cycle_end"]), "lengthDays": int(c["cycle_len"]),
             "anovulatory": bool(c["anovulatory"])}
        )
        if not c["anovulatory"]:
            # decode model ovulation within this cycle
            m = (g[C.KEY_DAY] >= c["cycle_start"]) & (g[C.KEY_DAY] < c["cycle_end"])
            cdays = g.loc[m, C.KEY_DAY].to_numpy()
            cprob = pred.phase_prob[g.loc[m].index.to_numpy(), C.PHASES.index("ovulation")]
            model_ov = int(cdays[int(np.nanargmax(np.nan_to_num(cprob)))]) if len(cdays) else -1
            out["events"].append(
                {"truthDay": int(c["true_ov_day"]), "calendarDay": int(c["cycle_start"] + 14),
                 "modelDay": model_ov}
            )
    return out


def main() -> None:
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / "participants").mkdir(exist_ok=True)

    df = pd.read_parquet("data/tier_c/cohort.parquet")
    # pick representative participants: a few regular, a few irregular, incl. hard cases
    reg = df.groupby(C.KEY_PARTICIPANT).apply(
        lambda g: (np.ptp(np.diff(g.loc[g.menses_reported == 1, C.KEY_DAY].to_numpy())) >= 9)
        if g.menses_reported.sum() > 2 else False, include_groups=False
    )
    irregular_pids = reg[reg].index.tolist()[:4]
    regular_pids = reg[~reg].index.tolist()[:3]
    chosen = regular_pids + irregular_pids
    sub = df[df[C.KEY_PARTICIPANT].isin(chosen)].copy()

    # Held-out predictions: train on everyone NOT chosen, predict the chosen participants.
    train_df = df[~df[C.KEY_PARTICIPANT].isin(chosen)].copy()
    pred = transfer_evaluate(train_df, sub, arm="wearable_menses")
    # residual std per hormone (empirical prediction error)
    resid_std = {}
    for h, arr in (("pdg", pred.pdg), ("e3g", pred.e3g)):
        r = pred.fm[h].to_numpy() - arr
        resid_std[h] = float(np.nanstd(r[np.isfinite(r)])) or 0.1

    index = []
    for pid, g in pred.fm.groupby(C.KEY_PARTICIPANT):
        g = g.sort_values(C.KEY_DAY)
        cyc = cycles_to_frame(extract_cycles(sub[sub[C.KEY_PARTICIPANT] == pid], use_metadata=True))
        payload = export_participant(pid, g, pred, resid_std, cyc)
        (WEB / "participants" / f"{pid}.json").write_text(json.dumps(payload))
        cvs = np.std(np.diff([c["startDay"] for c in payload["cycles"]])) if len(payload["cycles"]) > 1 else 0
        index.append({"pid": pid, "regularity": payload["regularity"],
                      "nCycles": len(payload["cycles"]), "cycleLenStd": _round(cvs, 1)})
    (WEB / "participants" / "index.json").write_text(json.dumps(index))

    # skill + leaderboard from committed results JSON
    export_skill_and_leaderboard()

    print(f"exported {len(index)} participants + skill/leaderboard to {WEB}")


def export_skill_and_leaderboard() -> None:
    def load(name):
        p = RESULTS / name
        return json.loads(p.read_text()) if p.exists() else None

    tierC = load("model_tierC.json")
    tierB = load("model_tierB.json")
    transfer = load("transfer_sim2real.json")

    def metric(res, task, stratum, metric_name):
        if not res:
            return None
        for m in res["metrics"]:
            if m["task"] == task and m["stratum"] == stratum and m["metric"] == metric_name:
                return m
        return None

    skill = {"strata": []}
    for stratum in ("all", "regular", "irregular"):
        c = metric(tierC, "T2-R", stratum, "SoC")
        b = metric(tierB, "T2-R", stratum, "SoC")
        skill["strata"].append({
            "stratum": stratum,
            "synthetic": {"soc": _round(c["value"]) if c else None,
                          "lo": _round(c["ci_lo"]) if c else None, "hi": _round(c["ci_hi"]) if c else None,
                          "n": c["n"] if c else 0, "p": _round(c["extra"].get("wilcoxon_p"), 4) if c else None,
                          "primary": c["is_primary"] if c else False},
            "real": {"soc": _round(b["value"]) if b else None,
                     "lo": _round(b["ci_lo"]) if b else None, "hi": _round(b["ci_hi"]) if b else None,
                     "n": b["n"] if b else 0, "p": _round(b["extra"].get("wilcoxon_p"), 4) if b else None,
                     "primary": b["is_primary"] if b else False},
        })
    (WEB / "skill.json").write_text(json.dumps(skill, indent=2))

    # leaderboard: task rows with synthetic vs real headline metrics
    rows = []
    def add(task, label, metric_name, fmt="{:.3f}"):
        c = metric(tierC, task, "all", metric_name)
        b = metric(tierB, task, "all", metric_name)
        t = metric(transfer, task, "all", metric_name)
        rows.append({"task": task, "label": label, "metric": metric_name,
                     "synthetic": _round(c["value"]) if c else None,
                     "real": _round(b["value"]) if b else None,
                     "transfer": _round(t["value"]) if t else None,
                     "n_real": b["n"] if b else 0})
    add("T3", "4-phase classification", "macro_f1")
    add("T2-R", "Ovulation timing (skill over calendar)", "SoC")
    add("T2-A", "Anovulation detection", "pr_auc")
    add("T1-pdg", "PdG reconstruction", "spearman")
    add("T1-e3g", "E3G reconstruction", "spearman")
    add("T1-NC", "Dilution negative control", "spearman_follicular_only")
    (WEB / "leaderboard.json").write_text(json.dumps(rows, indent=2))

    manifest = {
        "tiers": {"open": "NHANES/synthetic", "gated": "mcPHASES", "synthetic": "INFRADIAN-SYNTH-1K"},
        "real_cohort": {"n": 42, "note": "Canadian, ages 18-29 — says nothing about menopause/PCOS populations"},
        "headline": "On real consumer wearables the primary endpoint is null (p=0.48); the sim-to-real gap is the finding.",
    }
    (WEB / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
