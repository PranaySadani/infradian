"""Model + benchmark smoke tests: the pipeline runs, metrics are well-formed, and train/serve
feature parity holds (the same transform module is used for both, verified on a golden fixture)."""

from __future__ import annotations

import numpy as np
import pytest

from infradian.bench.model_runner import build_results, cross_validate_model
from infradian.features.build import build_features, feature_columns
from infradian.models.reference import train_bundle
from infradian.synth.generator import generate_cohort


@pytest.fixture(scope="module")
def cohort():
    return generate_cohort(n=48, n_days=120, seed=9)


def test_benchmark_runs_and_metrics_wellformed(cohort):
    pred = cross_validate_model(cohort, arm="wearable_menses")
    res = build_results(pred, cohort, tier="C", run_name="test", use_metadata=True)
    tasks = {m.task for m in res.metrics}
    assert "T3" in tasks and "T2-R" in tasks and "T1-pdg" in tasks
    for m in res.metrics:
        assert np.isfinite(m.value), f"{m.task}/{m.metric} produced a non-finite value"
        assert m.n > 0


def test_exactly_one_primary_endpoint_per_tier(cohort):
    pred = cross_validate_model(cohort, arm="wearable_menses")
    res = build_results(pred, cohort, tier="C", run_name="test", use_metadata=True)
    primaries = [m for m in res.metrics if m.is_primary]
    # The pre-registered primary is T2-R irregular SoC.
    assert len(primaries) == 1
    assert primaries[0].task == "T2-R" and primaries[0].stratum == "irregular"


def test_train_serve_feature_parity(cohort):
    """The bundle's feature columns must equal what build_features produces at serve time."""
    bundle = train_bundle(cohort, arm="wearable_menses")
    fm = build_features(cohort.head(200), arm="wearable_menses")
    serve_feats = feature_columns(fm)
    assert bundle["feature_columns"] == serve_feats, "train/serve feature columns diverged"


def test_t1_nc_control_is_small_relative_to_t1(cohort):
    """The follicular-only dilution control should be much weaker than the full T1 signal."""
    pred = cross_validate_model(cohort, arm="wearable_menses")
    res = build_results(pred, cohort, tier="C", run_name="test", use_metadata=True)
    by = {(m.task): m.value for m in res.metrics}
    if "T1-pdg" in by and "T1-NC" in by:
        assert by["T1-NC"] < by["T1-pdg"], "dilution control exceeds the hormone signal"
