"""Scoring: MAE, Skill over Calendar, cluster bootstrap, paired Wilcoxon.

Two rules from plan §8.3, both load-bearing:
  - SoC is computed on POOLED errors with a participant-macro weight, never as a mean of
    per-participant skill ratios (that ratio is the unstable quantity).
  - The confidence interval is a CLUSTER bootstrap over participants; the inferential test is a
    paired Wilcoxon on per-participant differences D_i. Neither is a t-test on ratios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon


def per_participant_mae(pid: np.ndarray, err: np.ndarray) -> dict[str, float]:
    """Mean absolute error per participant (err is already |pred - true|)."""
    out: dict[str, float] = {}
    for p in np.unique(pid):
        out[p] = float(np.mean(err[pid == p]))
    return out


def skill_over_calendar(model_mae_by_pid: dict[str, float], cal_mae_by_pid: dict[str, float]) -> float:
    """Participant-macro Skill over Calendar: 1 - (mean_i model_i) / (mean_i cal_i).

    Each participant contributes equally (macro), so a few high-volume participants can't dominate.
    """
    pids = sorted(set(model_mae_by_pid) & set(cal_mae_by_pid))
    if not pids:
        return float("nan")
    m = np.mean([model_mae_by_pid[p] for p in pids])
    c = np.mean([cal_mae_by_pid[p] for p in pids])
    return float(1.0 - m / c) if c > 0 else float("nan")


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    n: int


def cluster_bootstrap_skill(
    model_mae_by_pid: dict[str, float],
    cal_mae_by_pid: dict[str, float],
    b: int = 2000,
    seed: int = 0,
) -> Interval:
    """95% cluster-bootstrap CI for SoC, resampling PARTICIPANTS with replacement."""
    pids = np.array(sorted(set(model_mae_by_pid) & set(cal_mae_by_pid)))
    rng = np.random.default_rng(seed)
    point = skill_over_calendar(model_mae_by_pid, cal_mae_by_pid)
    if len(pids) < 2:
        return Interval(point, float("nan"), float("nan"), len(pids))
    stats = np.empty(b)
    m_arr = np.array([model_mae_by_pid[p] for p in pids])
    c_arr = np.array([cal_mae_by_pid[p] for p in pids])
    for i in range(b):
        idx = rng.integers(0, len(pids), len(pids))
        c_mean = c_arr[idx].mean()
        stats[i] = 1.0 - m_arr[idx].mean() / c_mean if c_mean > 0 else np.nan
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return Interval(point, float(lo), float(hi), len(pids))


@dataclass
class PairedTest:
    median_diff: float
    ci_lo: float
    ci_hi: float
    p_value: float
    n: int


def paired_difference_test(
    model_mae_by_pid: dict[str, float], cal_mae_by_pid: dict[str, float], seed: int = 0, b: int = 2000
) -> PairedTest:
    """Per-participant D_i = cal_MAE - model_MAE (days). Positive => model beats calendar.

    Returns median(D_i), a cluster-bootstrap CI for the median, and the Wilcoxon signed-rank p.
    This is the pre-registered primary inferential instrument.
    """
    pids = sorted(set(model_mae_by_pid) & set(cal_mae_by_pid))
    d = np.array([cal_mae_by_pid[p] - model_mae_by_pid[p] for p in pids])
    if len(d) < 2 or np.allclose(d, 0):
        return PairedTest(float(np.median(d)) if len(d) else 0.0, float("nan"), float("nan"), 1.0, len(d))
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(b)])
    lo, hi = np.percentile(meds, [2.5, 97.5])
    try:
        _, p = wilcoxon(d)
    except ValueError:
        p = 1.0
    return PairedTest(float(np.median(d)), float(lo), float(hi), float(p), len(d))
