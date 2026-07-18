"""The reference model bundle: train the three LightGBM models on a full cohort and save them with
their feature spec, so training and serving use identical transforms. The PUBLIC checkpoint
(`infradian-ref-s`) is trained on SYNTHETIC data only — the only checkpoint we may legally
distribute, which is precisely why sim-to-real transfer is a structural result, not a side note.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd

from infradian.data.canonical import FeatureSpec
from infradian.features.build import build_features, feature_columns
from infradian.models import gbm


def train_bundle(df: pd.DataFrame, arm: str = "wearable_menses") -> dict:
    """Train phase + PdG + E3G models on the full frame; return an in-memory bundle."""
    fm = build_features(df, arm=arm)
    feats = feature_columns(fm)
    X = fm[feats]
    bundle = {
        "arm": arm,
        "feature_columns": feats,
        "feature_spec": asdict(FeatureSpec()),
        "phase_clf": gbm.train_phase_classifier(X, fm["phase"]),
        "pdg_reg": gbm.train_hormone_regressor(X, fm["pdg"]),
        "e3g_reg": gbm.train_hormone_regressor(X, fm["e3g"]),
    }
    return bundle


def save_bundle(bundle: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_dir / "infradian_ref.joblib")
    spec = {
        "arm": bundle["arm"],
        "feature_columns": bundle["feature_columns"],
        "feature_spec": bundle["feature_spec"],
        "feature_columns_sha256": hashlib.sha256(
            json.dumps(bundle["feature_columns"], sort_keys=True).encode()
        ).hexdigest(),
    }
    (out_dir / "feature_spec.json").write_text(json.dumps(spec, indent=2))
    return out_dir / "infradian_ref.joblib"


def load_bundle(path: Path) -> dict:
    return joblib.load(path)
