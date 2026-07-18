"""Model evaluation: out-of-fold cross-validation of the reference model, Skill over Calendar,
the two pre-registered primary endpoints, and the sim-to-real transfer evaluation.

The reference model is trained on features and produces per-day phase probabilities and hormone
predictions. Ovulation is decoded per cycle from the ovulation-phase probability; anovulation is
scored from the peak ovulation probability (an ovulatory cycle has a concentrated thermal/HR
signature, an anovulatory one does not — the calendar has no analogue, which is why it cannot
compete on T2-A).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score

from infradian.bench import baselines as B
from infradian.bench import metrics as M
from infradian.bench.results import MetricBlock, RunResults, git_commit
from infradian.bench.splits import SEEDS, make_folds
from infradian.bench.tasks import cycles_to_frame, extract_cycles
from infradian.data import canonical as C
from infradian.features.build import build_features, feature_columns
from infradian.models import gbm

OV_IDX = C.PHASES.index("ovulation")


@dataclass
class ModelPredictions:
    fm: pd.DataFrame  # feature matrix (keys + features + targets)
    phase_prob: np.ndarray  # (n_rows, 4)
    pdg: np.ndarray  # (n_rows,)
    e3g: np.ndarray  # (n_rows,)


def _fit_predict(Xtr, fm_tr, Xte):
    """Train the three models on a fold and return (phase_prob, pdg_pred, e3g_pred) for the test."""
    clf = gbm.train_phase_classifier(Xtr, fm_tr["phase"])
    reg_pdg = gbm.train_hormone_regressor(Xtr, fm_tr["pdg"])
    reg_e3g = gbm.train_hormone_regressor(Xtr, fm_tr["e3g"])
    prob = clf.predict_proba(Xte)
    # LightGBM may see fewer than 4 classes in a fold; align columns to the full phase set.
    prob_full = np.zeros((len(Xte), len(C.PHASES)))
    for j, cls in enumerate(clf.classes_):
        prob_full[:, int(cls)] = prob[:, j]
    return prob_full, np.expm1(reg_pdg.predict(Xte)), np.expm1(reg_e3g.predict(Xte))


def cross_validate_model(df: pd.DataFrame, arm: str = "wearable_menses") -> ModelPredictions:
    """Within-tier OOF cross-validation. Each row is predicted only when its participant is held out."""
    fm = build_features(df, arm=arm)
    feats = feature_columns(fm)
    X = fm[feats]

    phase_sum = np.zeros((len(fm), len(C.PHASES)))
    pdg_sum = np.zeros(len(fm))
    e3g_sum = np.zeros(len(fm))
    count = np.zeros(len(fm))

    pid = fm[C.KEY_PARTICIPANT].to_numpy()
    for _seed, _fold, train_pids, test_pids in make_folds(df):
        tr = np.isin(pid, list(train_pids))
        te = np.isin(pid, list(test_pids))
        if te.sum() == 0 or tr.sum() == 0:
            continue
        prob, pdg_p, e3g_p = _fit_predict(X[tr], fm[tr], X[te])
        phase_sum[te] += prob
        pdg_sum[te] += pdg_p
        e3g_sum[te] += e3g_p
        count[te] += 1

    count = np.where(count == 0, np.nan, count)
    return ModelPredictions(
        fm=fm,
        phase_prob=phase_sum / count[:, None],
        pdg=pdg_sum / count,
        e3g=e3g_sum / count,
    )


def transfer_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, arm: str = "wearable_menses") -> ModelPredictions:
    """Train once on train_df (e.g. synthetic) and predict all of test_df (e.g. real). Sim-to-real."""
    fm_tr = build_features(train_df, arm=arm)
    fm_te = build_features(test_df, arm=arm)
    feats = [c for c in feature_columns(fm_tr) if c in feature_columns(fm_te)]
    prob, pdg_p, e3g_p = _fit_predict(fm_tr[feats], fm_tr, fm_te[feats])
    return ModelPredictions(fm=fm_te, phase_prob=prob, pdg=pdg_p, e3g=e3g_p)


# ---- metric assembly -------------------------------------------------------

def _decode_ovulation_by_cycle(pred: ModelPredictions, cycles: pd.DataFrame) -> pd.DataFrame:
    """For each cycle, decode the model's ovulation day and compute a thermal anovulation score.

    The anovulation score is the peak temperature CUSUM within the cycle: an ovulatory cycle has a
    sustained post-ovulation thermal elevation (high CUSUM), an anovulatory one does not. This needs
    no extra training and the calendar has no analogue.
    """
    fm = pred.fm
    ov_prob = pred.phase_prob[:, OV_IDX]
    lut_prob = pred.phase_prob[:, C.PHASES.index("luteal")]
    cusum_col = fm["skin_temp__cusum"] if "skin_temp__cusum" in fm.columns else None
    rows = []
    for _, cyc in cycles.iterrows():
        m = (
            (fm[C.KEY_SEGMENT] == cyc["segment_id"])
            & (fm[C.KEY_DAY] >= cyc["cycle_start"])
            & (fm[C.KEY_DAY] < cyc["cycle_end"])
        ).to_numpy()
        if m.sum() == 0:
            continue
        days = fm.loc[m, C.KEY_DAY].to_numpy()
        probs = np.nan_to_num(ov_prob[m])
        if probs.max() == 0 and np.all(np.isnan(ov_prob[m])):
            continue
        model_ov = gbm.decode_ovulation_day(days, probs)
        cusum_max = float(np.nanmax(cusum_col.to_numpy()[m])) if cusum_col is not None else 0.0
        # anovulation score: high when the ovulation+luteal thermal signature is WEAK
        thermal = float(np.nanmax(np.nan_to_num(ov_prob[m]) + np.nan_to_num(lut_prob[m])))
        rows.append(
            {
                "participant_id": cyc["participant_id"],
                "regularity": cyc["regularity"],
                "true_ov": cyc["true_ov_day"],
                "model_ov": model_ov,
                "anov_score": -(0.5 * cusum_max + thermal),  # low signature => high anovulation score
                "anovulatory": cyc["anovulatory"],
            }
        )
    return pd.DataFrame(rows)


def build_results(
    pred: ModelPredictions, df: pd.DataFrame, tier: str, run_name: str, use_metadata: bool
) -> RunResults:
    """Compute all task metrics from OOF predictions and write a results object."""
    t0 = time.time()
    fm = pred.fm
    cycles = cycles_to_frame(extract_cycles(df, use_metadata=use_metadata))
    decoded = _decode_ovulation_by_cycle(pred, cycles)

    res = RunResults(
        run_name=run_name,
        tier=tier,
        model="infradian-ref (LightGBM phase+hormone)",
        git_commit=git_commit(),
        seeds=list(SEEDS),
        dataset={
            "n_participants": int(df[C.KEY_PARTICIPANT].nunique()),
            "n_cycles": int(len(cycles)),
            "n_ovulatory": int((~cycles["anovulatory"]).sum()),
            "anovulatory_rate": float(cycles["anovulatory"].mean()),
        },
    )

    # ---- T3: 4-phase classification (macro-F1) ----
    true_phase = fm["phase"].map(gbm.PHASE_TO_INT)
    pred_phase = np.nanargmax(np.nan_to_num(pred.phase_prob, nan=-1), axis=1)
    mask = true_phase.notna().to_numpy() & np.isfinite(pred.phase_prob).any(axis=1)
    if mask.sum() > 0:
        f1 = f1_score(true_phase[mask].astype(int), pred_phase[mask], average="macro")
        res.add(MetricBlock(task="T3", metric="macro_f1", value=float(f1), n=int(mask.sum())))

    # ---- T2-R: model retrospective ovulation confirmation vs the PROSPECTIVE calendar (b1c).
    # This is the real-world comparison: a woman's period app shows a prospective calendar guess;
    # her watch instead confirms ovulation retrospectively. b1_retrospective is NOT a real baseline
    # because it needs the observed next onset, which she does not have when she wants to know.
    ovd = decoded[~decoded["anovulatory"]].copy()
    if len(ovd) > 0:
        ovul_cycles = cycles[~cycles["anovulatory"]].reset_index(drop=True)
        b1c_pred = B.b1c_calendar_hazard(ovul_cycles, ovul_cycles, prospective=True)
        cal_err_by_key = {
            (r["participant_id"], int(r["true_ov_day"])): abs(p - r["true_ov_day"])
            for (_, r), p in zip(ovul_cycles.iterrows(), b1c_pred, strict=True)
        }
        for stratum in ("all", "regular", "irregular"):
            sub = ovd if stratum == "all" else ovd[ovd["regularity"] == stratum]
            if len(sub) < 3:
                continue
            model_err = np.abs(sub["model_ov"].to_numpy() - sub["true_ov"].to_numpy())
            cal_err = np.array(
                [cal_err_by_key.get((r["participant_id"], int(r["true_ov"])), np.nan) for _, r in sub.iterrows()]
            )
            valid = np.isfinite(cal_err)
            pids = sub["participant_id"].to_numpy()[valid]
            m_by = M.per_participant_mae(pids, model_err[valid])
            c_by = M.per_participant_mae(pids, cal_err[valid])
            res.add(MetricBlock(task="T2-R", stratum=stratum, metric="mae_days",
                                value=float(np.mean(list(m_by.values()))), n=len(m_by),
                                extra={"model": True, "calendar_mae": float(np.mean(list(c_by.values())))}))
            soc = M.cluster_bootstrap_skill(m_by, c_by)
            pt = M.paired_difference_test(m_by, c_by)
            res.add(MetricBlock(task="T2-R", stratum=stratum, metric="SoC",
                                value=soc.point, ci_lo=soc.lo, ci_hi=soc.hi, n=soc.n,
                                is_primary=(stratum == "irregular"),
                                extra={"median_D_i": pt.median_diff, "wilcoxon_p": pt.p_value,
                                       "D_ci_lo": pt.ci_lo, "D_ci_hi": pt.ci_hi,
                                       "vs_baseline": "b1c_calendar_hazard (prospective)"}))

    # ---- T2-A: anovulation detection (PR-AUC vs prevalence). The calendar assumes ovulation every
    # cycle, so its PR-AUC is exactly the prevalence — it cannot compete by construction.
    if decoded["anovulatory"].nunique() == 2:
        score = decoded["anov_score"].to_numpy()
        y = decoded["anovulatory"].astype(int).to_numpy()
        ap = average_precision_score(y, score)
        res.add(MetricBlock(task="T2-A", metric="pr_auc", value=float(ap), n=len(y),
                            is_primary=False,
                            extra={"prevalence": float(y.mean()),
                                   "calendar_pr_auc": float(y.mean()),
                                   "lift_over_calendar": float(ap - y.mean())}))

    # ---- T1: hormone regression (within-participant Spearman) ----
    for hormone, preds in (("pdg", pred.pdg), ("e3g", pred.e3g)):
        rhos = []
        for _pid, g in fm.groupby(C.KEY_PARTICIPANT):
            idx = g.index
            yt = g[hormone].to_numpy()
            yp = preds[idx]
            ok = np.isfinite(yt) & np.isfinite(yp)
            if ok.sum() >= 8 and np.ptp(yt[ok]) > 0:
                rho = spearmanr(yt[ok], yp[ok]).statistic
                if np.isfinite(rho):
                    rhos.append(rho)
        if rhos:
            res.add(MetricBlock(task=f"T1-{hormone}", metric="spearman",
                                value=float(np.median(rhos)), n=len(rhos)))

    # ---- T1-NC: dilution negative control (PdG on follicular days only; should be ~0) ----
    fol = fm["phase"].isin(["menstruation", "late_follicular"]).to_numpy()
    rhos_nc = []
    for _pid, g in fm.groupby(C.KEY_PARTICIPANT):
        idx = g.index
        fmask = fol[idx]
        yt = g["pdg"].to_numpy()[fmask]
        yp = pred.pdg[idx][fmask]
        ok = np.isfinite(yt) & np.isfinite(yp)
        if ok.sum() >= 8 and np.ptp(yt[ok]) > 0:
            rho = spearmanr(yt[ok], yp[ok]).statistic
            if np.isfinite(rho):
                rhos_nc.append(rho)
    if rhos_nc:
        res.add(MetricBlock(task="T1-NC", metric="spearman_follicular_only",
                            value=float(np.median(rhos_nc)), n=len(rhos_nc),
                            extra={"interpretation": "dilution-artifact bound; expected ~0"}))

    res.wall_clock_s = round(time.time() - t0, 2)
    return res
