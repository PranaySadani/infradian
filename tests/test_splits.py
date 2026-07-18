"""Splits must be participant-disjoint and must never let a dual-round participant straddle folds."""

from __future__ import annotations

import pandas as pd
import pytest

from infradian.bench.splits import assert_participant_disjoint, make_folds, participant_regularity
from infradian.data import canonical as C
from infradian.synth.generator import generate_cohort


@pytest.fixture(scope="module")
def cohort():
    return generate_cohort(n=42, n_days=120, seed=11)


def test_no_participant_across_folds(cohort):
    assert_participant_disjoint(cohort)  # raises on violation


def test_every_participant_tested_once_per_repeat(cohort):
    n_pids = cohort[C.KEY_PARTICIPANT].nunique()
    from collections import Counter

    per_seed: dict[int, Counter] = {}
    for seed, _fold, _train, test in make_folds(cohort):
        per_seed.setdefault(seed, Counter()).update(test)
    for seed, counter in per_seed.items():
        assert len(counter) == n_pids, f"seed {seed}: not all participants tested"
        assert all(v == 1 for v in counter.values()), f"seed {seed}: a participant tested twice"


def test_dual_round_participant_grouped_by_participant_not_segment(cohort):
    """Simulate a participant with two rounds and assert both rounds land in the same fold."""
    # Take one participant, duplicate their data as a second round (segment r1) at day+905.
    pid = cohort[C.KEY_PARTICIPANT].iloc[0]
    p = cohort[cohort[C.KEY_PARTICIPANT] == pid].copy()
    p2 = p.copy()
    p2[C.KEY_SEGMENT] = C.make_segment_id(pid, 1)
    p2[C.KEY_DAY] = p2[C.KEY_DAY] + 905
    dual = pd.concat([cohort, p2], ignore_index=True)
    # The participant is grouped by participant_id, so both rounds land together — never split
    # by segment. assert_participant_disjoint raises if that invariant is violated.
    assert_participant_disjoint(dual)


def test_regularity_strata_present(cohort):
    reg = participant_regularity(cohort)
    vals = set(reg.values())
    assert "regular" in vals
    # with 42 participants at 33% irregular target, at least one irregular should appear
    assert "irregular" in vals
