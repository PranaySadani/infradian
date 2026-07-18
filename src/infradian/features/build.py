"""Build a causal feature matrix from the canonical long frame.

All features are computed per SEGMENT (participant-round), sorted by day, using only backward-looking
operators from `causal.py`. Segment grouping is what resets rolling windows at the mcPHASES round
boundary (~day 905) so a 14-day window never spans the two-year gap between Interval 1 and 2.

Feature arms (plan §6.4):
  - "wearable_menses": wearable signals + calendar features derived from observed menses onsets.
    This is the HEADLINE arm and the only fair comparison — it is exactly the information a
    period-tracking app has (a smartwatch plus the dates the user logged bleeding).
  - "wearable_only": wearable signals with no menses-onset information. Expected to be much weaker.
  - "diary": self-reported symptoms only, no wearable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from infradian.data import canonical as C
from infradian.features import causal
from infradian.features.registry import assert_feature_names_legal

WINDOWS = (3, 7, 14)
_WEARABLE_SIGNALS = ["skin_temp_dev_c", "rhr_bpm", "hrv_rmssd_ms", "resp_rate", "sleep_eff", "steps"]
_DIARY_SIGNALS = ["cramps", "mood", "stress"]


def _wearable_features(g: pd.DataFrame) -> dict[str, pd.Series]:
    """Causal wearable features for one segment (already sorted by day)."""
    feats: dict[str, pd.Series] = {}
    for sig in _WEARABLE_SIGNALS:
        s = g[sig]
        # personal-baseline z-score (causal expanding) — absolute values are cross-person meaningless
        feats[f"{sig}__z"] = causal.expanding_baseline_z(s)
        for w in WINDOWS:
            feats[f"{sig}__mean{w}"] = causal.causal_roll_mean(s, w)
            # rolling STD: the published self-report work found variability, not level, dominant
            feats[f"{sig}__std{w}"] = causal.causal_roll_std(s, w)
        feats[f"{sig}__delta1"] = causal.causal_delta(s, 1)
        feats[f"{sig}__delta7"] = causal.causal_delta(s, 7)
    # CUSUM on the two strongest luteal signals — short-memory, seasonal-drift-immune
    feats["skin_temp__cusum"] = causal.cusum_positive(g["skin_temp_dev_c"], drift=0.2)
    feats["rhr__cusum"] = causal.cusum_positive(g["rhr_bpm"], drift=0.2)
    return feats


def _menses_calendar_features(g: pd.DataFrame) -> dict[str, pd.Series]:
    """Calendar features derived from observed menses onsets (the 'app already knows this' info)."""
    dsl = causal.days_since_last_onset(g["menses_reported"])
    stats = causal.past_cycle_length_stats(g["menses_reported"])
    feats = {"days_since_last_onset": dsl}
    for col in stats.columns:
        feats[col] = stats[col]
    # phase-angle proxy: position within the expected cycle, using PAST median length only.
    expected = stats["cyclelen_median_past"].fillna(29.0)
    frac = (dsl.clip(lower=0) / expected).clip(0, 1.5)
    feats["cycle_frac_est"] = frac
    feats["cycle_frac_sin"] = np.sin(2 * np.pi * frac)
    feats["cycle_frac_cos"] = np.cos(2 * np.pi * frac)
    return feats


def _diary_features(g: pd.DataFrame) -> dict[str, pd.Series]:
    feats: dict[str, pd.Series] = {}
    for sig in _DIARY_SIGNALS:
        s = g[sig]
        for w in (3, 7):
            feats[f"{sig}__mean{w}"] = causal.causal_roll_mean(s, w)
            feats[f"{sig}__std{w}"] = causal.causal_roll_std(s, w)
    return feats


def build_features(df: pd.DataFrame, arm: str = "wearable_menses") -> pd.DataFrame:
    """Return a feature matrix: keys + features + targets/labels, one row per (segment, day).

    Targets/labels carried through for scoring: pdg, e3g (T1), phase (T3), and the metadata columns
    `_ovulation_day` / `_anovulatory` when present (T2 / T2-A). These are NOT features.
    """
    C.validate(df)
    out_frames = []
    for _, g in df.groupby(C.KEY_SEGMENT, sort=False):
        g = g.sort_values(C.KEY_DAY).reset_index(drop=True)
        feats: dict[str, pd.Series] = {}
        if arm in ("wearable_menses", "wearable_only"):
            feats.update(_wearable_features(g))
        if arm == "wearable_menses":
            feats.update(_menses_calendar_features(g))
        if arm == "diary":
            feats.update(_diary_features(g))
            feats.update(_menses_calendar_features(g))
        if not feats:
            raise ValueError(f"unknown feature arm: {arm}")

        fmat = pd.DataFrame(feats, index=g.index)
        assert_feature_names_legal(list(fmat.columns))

        keep_keys = g[[C.KEY_PARTICIPANT, C.KEY_SEGMENT, C.KEY_DAY]].copy()
        targets = g[["pdg", "e3g", "lh", "phase", "menses_reported"]].copy()
        meta_cols = [c for c in ("_ovulation_day", "_anovulatory", "_cycle_start_day") if c in g.columns]
        meta = g[meta_cols].copy() if meta_cols else pd.DataFrame(index=g.index)

        out_frames.append(pd.concat([keep_keys, fmat, targets, meta], axis=1))

    result = pd.concat(out_frames, ignore_index=True)
    return result


def feature_columns(feature_matrix: pd.DataFrame) -> list[str]:
    """The model-input columns: everything that is not a key, target, label, or metadata."""
    non_features = {
        C.KEY_PARTICIPANT,
        C.KEY_SEGMENT,
        C.KEY_DAY,
        "pdg",
        "e3g",
        "lh",
        "phase",
        "menses_reported",
        "_ovulation_day",
        "_anovulatory",
        "_cycle_start_day",
    }
    return [c for c in feature_matrix.columns if c not in non_features]
