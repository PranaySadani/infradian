"""LightGBM reference models. Deliberately shallow and strongly regularized — with n=42 real
participants, overfitting is the dominant risk. Hyperparameters are fixed (tuned on synthetic
only, never on the evaluation data) so there is zero selection leakage.

LightGBM handles NaN natively, which matters because wearable channels have real gaps.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from infradian.data import canonical as C

# Fixed, conservative hyperparameters (tuned on synthetic; frozen for all real-data evaluation).
_COMMON = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,  # shallow — guards against n=42 overfit
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=0,
    verbose=-1,
    n_jobs=1,
)

PHASE_TO_INT = {p: i for i, p in enumerate(C.PHASES)}
INT_TO_PHASE = {i: p for p, i in PHASE_TO_INT.items()}


def train_phase_classifier(X: pd.DataFrame, y_phase: pd.Series) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=len(C.PHASES), **_COMMON)
    y = y_phase.map(PHASE_TO_INT)
    mask = y.notna()
    clf.fit(X[mask], y[mask].astype(int))
    return clf


def train_hormone_regressor(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
    reg = lgb.LGBMRegressor(objective="regression_l1", **_COMMON)
    mask = y.notna()
    reg.fit(X[mask], np.log1p(y[mask].clip(lower=0)))
    return reg


def train_anovulation_classifier(Xc: pd.DataFrame, y_anov: np.ndarray) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(objective="binary", **{**_COMMON, "num_leaves": 7, "n_estimators": 150})
    clf.fit(Xc, y_anov.astype(int))
    return clf


def decode_ovulation_day(days: np.ndarray, ovulation_prob: np.ndarray, smooth: int = 3) -> int:
    """Median-filter the per-day ovulation probability and take the single argmax day.

    This is the decoder ablation's 'constrained' arm: a raw per-day classifier emits several
    ovulation-labelled days per cycle; median smoothing + one argmax yields exactly one.
    """
    if len(ovulation_prob) == 0:
        return int(days[0]) if len(days) else -1
    p = pd.Series(ovulation_prob).rolling(smooth, min_periods=1, center=True).median().to_numpy()
    return int(days[int(np.argmax(p))])


def cycle_level_features(feat_rows: pd.DataFrame, ovulation_prob: np.ndarray) -> dict[str, float]:
    """Aggregate a cycle's per-day features into cycle-level features for anovulation detection.

    The key signal: an ovulatory cycle has a sustained thermal/HR elevation (a high, concentrated
    ovulation/luteal probability); an anovulatory cycle does not. The calendar has no analogue.
    """
    temp = feat_rows.get("skin_temp_dev_c__z")
    rhr = feat_rows.get("rhr_bpm__z")
    cusum = feat_rows.get("skin_temp__cusum")
    return {
        "max_ov_prob": float(np.nanmax(ovulation_prob)) if len(ovulation_prob) else 0.0,
        "mean_ov_prob": float(np.nanmean(ovulation_prob)) if len(ovulation_prob) else 0.0,
        "temp_range": float(np.nanmax(temp) - np.nanmin(temp)) if temp is not None and temp.notna().any() else 0.0,
        "temp_late_mean": float(np.nanmean(temp.to_numpy()[-10:])) if temp is not None and temp.notna().any() else 0.0,
        "rhr_range": float(np.nanmax(rhr) - np.nanmin(rhr)) if rhr is not None and rhr.notna().any() else 0.0,
        "cusum_max": float(np.nanmax(cusum)) if cusum is not None and cusum.notna().any() else 0.0,
    }
