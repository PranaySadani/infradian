"""Causal windowing primitives. EVERYTHING that touches a rolling/expanding statistic goes
through this module. Nothing else in the package may call `.rolling()`, `.expanding()`, or
`.shift()` directly.

The contract, enforced by tests/test_feature_causality.py:

  For every feature value at day t, the value depends ONLY on data at days <= t.
  Equivalently: truncating the series at t and recomputing gives an identical value at t
  (truncation invariance), and mutating any day > t leaves the value at t unchanged
  (future-perturbation invariance).

This is what makes the benchmark's leakage claims mechanically true rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_roll_mean(s: pd.Series, window: int) -> pd.Series:
    """Trailing rolling mean over the last `window` days (inclusive of today)."""
    return s.rolling(window=window, min_periods=1).mean()


def causal_roll_std(s: pd.Series, window: int) -> pd.Series:
    """Trailing rolling std (population-ish, min_periods=2 so a single point is NaN->0)."""
    return s.rolling(window=window, min_periods=2).std().fillna(0.0)


def causal_delta(s: pd.Series, lag: int) -> pd.Series:
    """Difference between today and `lag` days ago. Uses only the past."""
    return s - s.shift(lag)


def expanding_baseline_z(s: pd.Series, min_periods: int = 5) -> pd.Series:
    """Personal-baseline z-score using an expanding (causal) mean/std up to and including today.

    This replaces the naive whole-series z-score, which would leak the future. Absolute skin
    temperature and RHR are meaningless across people, so this personal normalization is mandatory;
    doing it causally is what keeps it honest.
    """
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    z = (s - mu) / sd.replace(0.0, np.nan)
    return z.fillna(0.0)


def days_since_last_onset(menses_reported: pd.Series) -> pd.Series:
    """Days since the last OBSERVED menses onset (a menses day preceded by a non-menses day).

    Only past onsets count. NaN (encoded as -1) before the first observed onset. This is the ONE
    legal cycle-position feature; `cycle_day` derived from the label is banned (see registry).
    """
    m = menses_reported.fillna(0).to_numpy().astype(int)
    onset = np.zeros(len(m), dtype=bool)
    for i in range(len(m)):
        if m[i] == 1 and (i == 0 or m[i - 1] == 0):
            onset[i] = True
    out = np.full(len(m), -1, dtype=float)
    last = None
    for i in range(len(m)):
        if last is not None:
            out[i] = i - last
        if onset[i]:
            last = i
            out[i] = 0
    return pd.Series(out, index=menses_reported.index)


def past_cycle_length_stats(menses_reported: pd.Series) -> pd.DataFrame:
    """Causal running statistics of COMPLETED cycle lengths (median, last, count) up to each day.

    A cycle length is the gap between consecutive observed onsets; it becomes known only at the
    second onset, so at day t we use only cycles whose closing onset is <= t. This is the
    information a period-tracking app actually has, and it feeds the calendar baselines.
    """
    m = menses_reported.fillna(0).to_numpy().astype(int)
    onset_days = [
        i for i in range(len(m)) if m[i] == 1 and (i == 0 or m[i - 1] == 0)
    ]
    median = np.full(len(m), np.nan)
    last = np.full(len(m), np.nan)
    count = np.zeros(len(m))

    completed: list[int] = []
    oi = 0  # index into onset_days
    for i in range(len(m)):
        # absorb any onsets that closed on or before day i
        while oi < len(onset_days) and onset_days[oi] <= i:
            if oi >= 1:
                completed.append(onset_days[oi] - onset_days[oi - 1])
            oi += 1
        if completed:
            median[i] = float(np.median(completed))
            last[i] = float(completed[-1])
            count[i] = len(completed)
    return pd.DataFrame(
        {
            "cyclelen_median_past": median,
            "cyclelen_last_past": last,
            "cyclelen_count_past": count,
        },
        index=menses_reported.index,
    )


def cusum_positive(s: pd.Series, drift: float = 0.0) -> pd.Series:
    """Causal one-sided (positive) CUSUM of the personal-baseline-centered series.

    Accumulates positive deviations from an expanding baseline; resets at zero. Captures the onset
    of a sustained temperature/HR elevation (the luteal shift) without a long lookback window, so it
    is largely immune to the seasonal drift confound that exposes 30-day baseline-deviation features.
    """
    z = expanding_baseline_z(s)
    out = np.zeros(len(z))
    acc = 0.0
    vals = z.to_numpy()
    for i in range(len(vals)):
        acc = max(0.0, acc + vals[i] - drift)
        out[i] = acc
    return pd.Series(out, index=s.index)
