"""Mechanical proof that no feature leaks the future.

Two independent checks, applied to the whole feature matrix:

  1. Future-perturbation: mutate every data value AFTER day t, recompute, and assert the feature
     row at day t is byte-for-byte unchanged.
  2. Truncation-invariance: compute features on the series truncated at t, and assert the value at
     t equals the value computed on the full series.

If either fails, some feature is peeking ahead. This is the test that makes the benchmark's
leakage claims true rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from infradian.features.build import build_features, feature_columns
from infradian.features.registry import assert_feature_names_legal
from infradian.synth.generator import generate_cohort

ARMS = ["wearable_menses", "wearable_only", "diary"]


@pytest.fixture(scope="module")
def small_cohort():
    # A handful of participants is enough; keep it fast.
    return generate_cohort(n=6, n_days=120, seed=3)


@pytest.mark.parametrize("arm", ARMS)
def test_no_banned_feature_names(small_cohort, arm):
    fm = build_features(small_cohort, arm=arm)
    assert_feature_names_legal(feature_columns(fm))  # raises if any leak


@pytest.mark.parametrize("arm", ARMS)
def test_future_perturbation_invariance(small_cohort, arm):
    """Corrupting the future must not change any feature at day t."""
    rng = np.random.default_rng(0)
    # Pick one segment to perturb.
    seg = small_cohort["segment_id"].iloc[0]
    seg_df = small_cohort[small_cohort["segment_id"] == seg].sort_values("day_in_study").reset_index(drop=True)
    t = len(seg_df) // 2

    base = build_features(seg_df, arm=arm).sort_values("day_in_study").reset_index(drop=True)

    corrupted = seg_df.copy()
    numeric = corrupted.select_dtypes(include="number").columns
    corrupted[numeric] = corrupted[numeric].astype(float)  # allow float noise on int columns
    fut = corrupted.index[corrupted["day_in_study"] > seg_df["day_in_study"].iloc[t]]
    corrupted.loc[fut, numeric] = corrupted.loc[fut, numeric] + rng.normal(50, 50, size=(len(fut), len(numeric)))
    # also flip menses in the future
    corrupted.loc[fut, "menses_reported"] = 1.0 - corrupted.loc[fut, "menses_reported"]
    # keep day_in_study intact (it was in `numeric`)
    corrupted["day_in_study"] = seg_df["day_in_study"].values

    pert = build_features(corrupted, arm=arm).sort_values("day_in_study").reset_index(drop=True)

    cols = feature_columns(base)
    row_base = base.loc[base["day_in_study"] == seg_df["day_in_study"].iloc[t], cols].to_numpy()
    row_pert = pert.loc[pert["day_in_study"] == seg_df["day_in_study"].iloc[t], cols].to_numpy()
    np.testing.assert_allclose(
        np.nan_to_num(row_base), np.nan_to_num(row_pert), rtol=1e-9, atol=1e-9,
        err_msg=f"arm {arm}: feature at day t changed when the future was corrupted",
    )


@pytest.mark.parametrize("arm", ARMS)
def test_truncation_invariance(small_cohort, arm):
    """Features at day t computed on the truncated series equal those on the full series."""
    seg = small_cohort["segment_id"].iloc[0]
    seg_df = small_cohort[small_cohort["segment_id"] == seg].sort_values("day_in_study").reset_index(drop=True)
    t_day = seg_df["day_in_study"].iloc[len(seg_df) * 2 // 3]

    full = build_features(seg_df, arm=arm)
    trunc = build_features(seg_df[seg_df["day_in_study"] <= t_day].copy(), arm=arm)

    cols = feature_columns(full)
    row_full = full.loc[full["day_in_study"] == t_day, cols].to_numpy()
    row_trunc = trunc.loc[trunc["day_in_study"] == t_day, cols].to_numpy()
    np.testing.assert_allclose(
        np.nan_to_num(row_full), np.nan_to_num(row_trunc), rtol=1e-9, atol=1e-9,
        err_msg=f"arm {arm}: feature at day t differs between full and truncated series",
    )


def test_wearable_only_excludes_menses_features(small_cohort):
    fm = build_features(small_cohort, arm="wearable_only")
    cols = feature_columns(fm)
    assert not any("days_since_last_onset" in c or "cycle_frac" in c for c in cols)


def test_wearable_menses_includes_calendar_features(small_cohort):
    fm = build_features(small_cohort, arm="wearable_menses")
    cols = feature_columns(fm)
    assert "days_since_last_onset" in cols
    assert "cyclelen_median_past" in cols
