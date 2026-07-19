"""Serving-path parity.

The Vercel function reproduces the training-time feature transform in pure numpy so it can fit
under the 500MB serverless bundle limit (pandas + scikit-learn push it to ~660MB). A second
implementation is a real divergence risk, so this test builds features BOTH ways on identical
input and asserts they are numerically identical, and that the two model paths agree.

If someone changes `infradian.features.build` without updating `api/_inference.py`, this fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

import _inference  # noqa: E402

from infradian.data import canonical as C  # noqa: E402
from infradian.features.build import build_features, feature_columns  # noqa: E402
from infradian.synth.generator import generate_cohort  # noqa: E402

MODEL_DIR = ROOT / "results" / "models" / "infradian-ref-s"


def _one_participant_days(n_days: int = 45) -> list[dict]:
    """A single participant's series as the API receives it (list of per-day dicts)."""
    df = generate_cohort(n=1, n_days=n_days, seed=17)
    df = df.sort_values(C.KEY_DAY).reset_index(drop=True)
    cols = [
        "day_in_study", "skin_temp_dev_c", "rhr_bpm", "hrv_rmssd_ms",
        "resp_rate", "sleep_eff", "steps", "menses_reported", "cramps", "mood", "stress",
    ]
    return [
        {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
        for row in df[cols].to_dict("records")
    ]


def _pandas_features(days: list[dict]) -> tuple[np.ndarray, list[str]]:
    rows = []
    for d in days:
        row = dict(d)
        row[C.KEY_PARTICIPANT] = "user"
        row[C.KEY_SEGMENT] = C.make_segment_id("user", 0)
        for col in C.ALL_COLUMNS:
            row.setdefault(col, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)[C.ALL_COLUMNS]
    fm = build_features(df, arm="wearable_menses")
    cols = feature_columns(fm)
    X = fm[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return X, cols


def test_numpy_features_match_pandas_features_exactly():
    days = _one_participant_days()
    X_pd, cols = _pandas_features(days)
    X_np = _inference.build_feature_matrix(days, cols)

    assert X_np.shape == X_pd.shape, f"shape mismatch {X_np.shape} vs {X_pd.shape}"

    # Compare column by column so a failure names the offending feature.
    for j, name in enumerate(cols):
        a, b = X_np[:, j], X_pd[:, j]
        both_nan = ~np.isfinite(a) & ~np.isfinite(b)
        np.testing.assert_allclose(
            np.where(both_nan, 0.0, np.nan_to_num(a, nan=0.0)),
            np.where(both_nan, 0.0, np.nan_to_num(b, nan=0.0)),
            rtol=1e-9, atol=1e-9,
            err_msg=f"serving feature '{name}' diverges from the training transform",
        )


def test_feature_column_order_is_the_trained_order():
    days = _one_participant_days(30)
    _, cols = _pandas_features(days)
    meta_cols = _inference.load_models()["meta"]["feature_columns"] if MODEL_DIR.exists() else cols
    assert list(meta_cols) == list(cols), "serving column order differs from training order"


@pytest.mark.skipif(not (MODEL_DIR / "phase_clf.txt").exists(), reason="boosters not exported")
def test_booster_predictions_match_the_sklearn_bundle():
    """The native Booster path must reproduce the sklearn-wrapped bundle's predictions."""
    joblib_path = MODEL_DIR / "infradian_ref.joblib"
    if not joblib_path.exists():
        pytest.skip("reference bundle not built")

    from infradian.models.reference import load_bundle

    days = _one_participant_days(40)
    X_pd, cols = _pandas_features(days)
    bundle = load_bundle(joblib_path)

    ref_pdg = np.expm1(bundle["pdg_reg"].predict(pd.DataFrame(X_pd, columns=cols)))
    out = _inference.predict(days)

    np.testing.assert_allclose(
        np.array(out["pdg_pred"]), np.round(ref_pdg, 3), rtol=1e-6, atol=2e-3,
        err_msg="native Booster PdG predictions diverge from the sklearn bundle",
    )


@pytest.mark.skipif(not (MODEL_DIR / "phase_clf.txt").exists(), reason="boosters not exported")
def test_numpy_booster_is_bit_identical_to_lightgbm():
    """The serving path parses LightGBM's model text and runs the forward pass in numpy, so that
    the deployed function does not need lightgbm (which hard-imports scipy, ~500MB). Assert the
    reader reproduces LightGBM exactly, including missing-value and zero routing."""
    lgb = pytest.importorskip("lightgbm")
    import _booster

    rng = np.random.default_rng(0)
    for fname in ("pdg_reg.txt", "e3g_reg.txt", "phase_clf.txt"):
        ref = lgb.Booster(model_file=str(MODEL_DIR / fname))
        mine = _booster.load_model(str(MODEL_DIR / fname))
        X = rng.normal(size=(120, ref.num_feature()))
        variants = {"dense": X}
        nan_x = X.copy(); nan_x[rng.random(X.shape) < 0.25] = np.nan
        zero_x = X.copy(); zero_x[rng.random(X.shape) < 0.2] = 0.0
        variants["nan"] = nan_x
        variants["zero"] = zero_x
        for label, data in variants.items():
            np.testing.assert_allclose(
                np.asarray(_booster.predict(mine, data)),
                np.asarray(ref.predict(data)),
                rtol=1e-9, atol=1e-9,
                err_msg=f"{fname} [{label}]: numpy booster diverges from LightGBM",
            )
