"""Benchmark runner. At CP1 this produces a publishable results file containing the calendar
baselines alone (b1a/b1b/b1c) on the ovulation-timing tasks, proving the harness end to end. At
CP3 the reference model is added and Skill over Calendar (model vs b1c) is computed, including the
two pre-registered primary endpoints.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from infradian.bench import baselines as B
from infradian.bench import metrics as M
from infradian.bench.results import MetricBlock, RunResults, git_commit
from infradian.bench.splits import SEEDS, make_folds
from infradian.bench.tasks import cycles_to_frame, extract_cycles

# A predictor takes (train_cycles, test_cycles, all_train_history) -> ov predictions for test.
Predictor = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], np.ndarray]


def _oof_predict(df: pd.DataFrame, cyc: pd.DataFrame, predictor: Predictor) -> np.ndarray:
    """Out-of-fold predictions for every cycle, averaged across repeat seeds."""
    acc = np.zeros((len(cyc), len(SEEDS)))
    acc[:] = np.nan
    cyc = cyc.reset_index(drop=True)
    for seed, _fold, train_pids, test_pids in _grouped_folds(df):
        train_cyc = cyc[cyc["participant_id"].isin(train_pids)]
        test_mask = cyc["participant_id"].isin(test_pids)
        test_cyc = cyc[test_mask]
        if len(test_cyc) == 0:
            continue
        preds = predictor(train_cyc, test_cyc, train_cyc)
        acc[test_cyc.index.to_numpy(), _seed_col(seed)] = preds
    return np.nanmean(acc, axis=1)


# helper to map seed -> column and iterate folds keeping seed grouping
def _seed_col(seed: int) -> int:
    return SEEDS.index(seed)


def _grouped_folds(df: pd.DataFrame):
    yield from make_folds(df)


def _mae_by_pid(cyc: pd.DataFrame, preds: np.ndarray) -> dict[str, float]:
    err = np.abs(preds - cyc["true_ov_day"].to_numpy())
    return M.per_participant_mae(cyc["participant_id"].to_numpy(), err)


def _mae_by_pid_stratified(cyc: pd.DataFrame, preds: np.ndarray, stratum: str) -> dict[str, float]:
    if stratum == "all":
        sub = cyc
        sub_preds = preds
    else:
        mask = (cyc["regularity"] == stratum).to_numpy()
        sub = cyc[mask]
        sub_preds = preds[mask]
    return _mae_by_pid(sub, sub_preds)


def run_baselines(df: pd.DataFrame, tier: str, run_name: str = "baselines") -> RunResults:
    """CP1 deliverable: baseline MAE on T2-R and T2-P, overall and by regularity stratum."""
    t0 = time.time()
    cyc = cycles_to_frame(extract_cycles(df))
    ovul = cyc[~cyc["anovulatory"]].reset_index(drop=True)

    res = RunResults(
        run_name=run_name,
        tier=tier,
        model="calendar-baselines",
        git_commit=git_commit(),
        seeds=list(SEEDS),
        dataset={
            "n_participants": int(df["participant_id"].nunique()),
            "n_cycles": int(len(cyc)),
            "n_ovulatory_cycles": int(len(ovul)),
            "anovulatory_rate": float((cyc["anovulatory"]).mean()),
        },
        splits={"scheme": "RepeatedStratifiedGroupKFold", "n_splits": 6, "n_repeats": len(SEEDS)},
    )

    # --- T2-R retrospective ---
    retro_preds = {
        "b1a_fixed14": B.b1a_fixed14(ovul),
        "b1b_population": _oof_predict(df, ovul, lambda tr, te, hist: B.b1b_population(te, tr)),
        "b1c_retro": B.b1_retrospective(ovul),
    }
    for name, preds in retro_preds.items():
        for stratum in ("all", "regular", "irregular"):
            mae = _mae_by_pid_stratified(ovul, preds, stratum)
            if not mae:
                continue
            res.add(
                MetricBlock(
                    task="T2-R",
                    stratum=stratum,
                    metric="mae_days",
                    value=float(np.mean(list(mae.values()))),
                    n=len(mae),
                    extra={"baseline": name},
                )
            )

    # --- T2-P prospective (b1c hazard is the reference denominator) ---
    prosp_preds = {
        "b1a_fixed14": B.b1a_fixed14(ovul),
        "b1b_population": _oof_predict(df, ovul, lambda tr, te, hist: B.b1b_population(te, tr)),
        "b1c_hazard": _oof_predict(
            df, ovul, lambda tr, te, hist: B.b1c_calendar_hazard(te, hist, prospective=False)
        ),
    }
    for name, preds in prosp_preds.items():
        for stratum in ("all", "regular", "irregular"):
            mae = _mae_by_pid_stratified(ovul, preds, stratum)
            if not mae:
                continue
            res.add(
                MetricBlock(
                    task="T2-P",
                    stratum=stratum,
                    metric="mae_days",
                    value=float(np.mean(list(mae.values()))),
                    n=len(mae),
                    extra={"baseline": name},
                )
            )

    res.wall_clock_s = round(time.time() - t0, 2)
    res.notes = "CP1: calendar baselines only. b1c is the reference denominator for Skill over Calendar."
    return res
