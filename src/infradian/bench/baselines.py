"""Calendar baselines for ovulation-day prediction — the denominator of Skill over Calendar.

The whole benchmark hinges on these being STRONG, not strawmen. The headline denominator is
`b1c_calendar_hazard`, an empirical-Bayes per-participant cycle-length model with a conditional
hazard update — precisely the adaptive behaviour a good period-tracking app should have. We report
the weaker baselines too, only to show the gap.

Retrospective (T2-R): the full cycle is observed, so cycle length is known and the calendar can
place ovulation a fixed luteal length before the next onset.
Prospective (T2-P): at day t the next onset is unknown, so the calendar must forecast cycle length.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ASSUMED_LUTEAL = 14  # classic fixed luteal phase length (days before next menses)


def _population_median_cycle(train_cycles: pd.DataFrame) -> float:
    lens = train_cycles["cycle_len"].to_numpy()
    return float(np.median(lens)) if len(lens) else 29.0


def b1a_fixed14(cycles: pd.DataFrame) -> np.ndarray:
    """Fixed cycle-day-14 ovulation (ov = onset + 13). A deliberate strawman for contrast."""
    return cycles["cycle_start"].to_numpy() + 13


def b1b_population(cycles: pd.DataFrame, train_cycles: pd.DataFrame) -> np.ndarray:
    """Population-template: ov = onset + (population median cycle length - assumed luteal).

    Non-adaptive within a cycle: commits at onset, never updates."""
    med = _population_median_cycle(train_cycles)
    return cycles["cycle_start"].to_numpy() + (med - ASSUMED_LUTEAL)


def b1c_calendar_hazard(
    cycles: pd.DataFrame, history: pd.DataFrame, prospective: bool = False
) -> np.ndarray:
    """Empirical-Bayes per-participant cycle-length model with shrinkage toward population.

    For each cycle, the expected length is a precision-weighted blend of the participant's own past
    cycle lengths and the population mean; ovulation = onset + (expected_length - assumed luteal).
    Participants with few observed cycles shrink toward the population (which is how this baseline
    stays strong for irregular participants with sparse history).

    `history` supplies the cycles available for estimating personal means (the training cycles).
    `prospective=True` uses only cycles that started strictly before each target cycle.
    """
    pop_mean = float(np.mean(history["cycle_len"])) if len(history) else 29.0
    pop_var = float(np.var(history["cycle_len"])) if len(history) > 1 else 16.0
    pop_var = max(pop_var, 1e-6)

    preds = np.empty(len(cycles))
    for j, (_, row) in enumerate(cycles.reset_index(drop=True).iterrows()):
        pid = row["participant_id"]
        past = history[history["participant_id"] == pid]
        if prospective:
            past = past[past["cycle_start"] < row["cycle_start"]]
        lens = past["cycle_len"].to_numpy()
        if len(lens) == 0:
            expected = pop_mean
        else:
            # empirical-Bayes shrinkage: within-participant precision vs population precision
            personal_mean = float(np.mean(lens))
            within_var = float(np.var(lens)) if len(lens) > 1 else pop_var
            within_var = max(within_var, 1e-6)
            w_personal = len(lens) / within_var
            w_pop = 1.0 / pop_var
            expected = (w_personal * personal_mean + w_pop * pop_mean) / (w_personal + w_pop)
        preds[j] = row["cycle_start"] + (expected - ASSUMED_LUTEAL)
    return preds


def b1_retrospective(cycles: pd.DataFrame) -> np.ndarray:
    """Strongest retrospective calendar: cycle length is fully observed, so place ovulation a
    fixed luteal length before the observed next onset. This is the fair T2-R baseline."""
    return cycles["cycle_end"].to_numpy() - ASSUMED_LUTEAL
