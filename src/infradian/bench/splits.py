"""Participant-disjoint cross-validation splits.

Repeated Stratified Group K-Fold: grouped on `participant_id` (never `segment_id`, so the 20
dual-round mcPHASES participants never straddle a fold), stratified on cycle regularity (so the
~10-14 irregular participants are spread across folds rather than piling into one).

Not a single holdout (at n=42 the metric standard error is uninterpretable) and not LOSO (per-fold
metrics become useless and the skill denominator goes per-participant, the unstable quantity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from infradian.data import canonical as C

N_SPLITS = 6
SEEDS = (0, 1)  # 2 repeats


def regularity_of(cycle_lengths: np.ndarray) -> str:
    """Frozen operational definition (plan §8.1): irregular if the cycle-length range is >= 9 days
    OR any cycle is shorter than 24 or longer than 38 days."""
    if len(cycle_lengths) == 0:
        return "regular"
    rng = float(np.ptp(cycle_lengths))
    extreme = bool(((cycle_lengths < 24) | (cycle_lengths > 38)).any())
    return "irregular" if (rng >= 9 or extreme) else "regular"


def participant_regularity(df: pd.DataFrame) -> dict[str, str]:
    """Map each participant to its regularity stratum from observed menses onsets."""
    out: dict[str, str] = {}
    for pid, g in df.groupby(C.KEY_PARTICIPANT):
        m = g.sort_values(C.KEY_DAY)["menses_reported"].fillna(0).to_numpy().astype(int)
        days = g.sort_values(C.KEY_DAY)[C.KEY_DAY].to_numpy()
        onsets = [days[i] for i in range(len(m)) if m[i] == 1 and (i == 0 or m[i - 1] == 0)]
        lens = np.diff(onsets) if len(onsets) >= 2 else np.array([])
        out[pid] = regularity_of(lens)
    return out


def make_folds(df: pd.DataFrame, seeds: tuple[int, ...] = SEEDS, n_splits: int = N_SPLITS):
    """Yield (repeat_seed, fold_idx, train_pids, test_pids) tuples.

    Splitting happens at the PARTICIPANT level; callers map rows to folds by participant.
    """
    reg = participant_regularity(df)
    pids = np.array(sorted(reg.keys()))
    strat = np.array([reg[p] for p in pids])
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # groups == pids so each participant is a single unit; X is a dummy.
        for fold_idx, (tr, te) in enumerate(sgkf.split(pids, strat, groups=pids)):
            yield seed, fold_idx, set(pids[tr].tolist()), set(pids[te].tolist())


def assert_participant_disjoint(df: pd.DataFrame) -> None:
    """Assert no participant appears in more than one test fold within a repeat. Used by tests."""
    for seed in SEEDS:
        seen: dict[str, int] = {}
        for s, fold_idx, _train, test in make_folds(df, seeds=(seed,)):
            for pid in test:
                if pid in seen:
                    raise AssertionError(
                        f"participant {pid} appears in test folds {seen[pid]} and {fold_idx} "
                        f"within repeat seed={s}"
                    )
                seen[pid] = fold_idx
