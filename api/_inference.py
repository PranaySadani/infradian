"""Dependency-light inference path for the Vercel serverless function.

Why this exists: the training-time feature builder uses pandas, and the model bundle is a
scikit-learn-wrapped LightGBM estimator. Together with scipy that is ~660MB installed, which
exceeds Vercel's 500MB function limit. This module reproduces the exact same feature vector using
numpy alone, and loads the models from LightGBM's native Booster text format, so the deployed
function needs only numpy, scipy (a hard lightgbm dependency) and lightgbm.

The obvious risk of a second implementation is silent divergence from the trained-on transform.
`tests/test_serve_parity.py` guards against that: it builds features both ways on the same input
and asserts they are numerically identical, so this file cannot drift from
`infradian.features.build` without the test suite failing.

Mirrors `build_features(arm="wearable_menses")` and the causal helpers in
`infradian.features.causal`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "results" / "models" / "infradian-ref-s"

WEARABLE_SIGNALS = ["skin_temp_dev_c", "rhr_bpm", "hrv_rmssd_ms", "resp_rate", "sleep_eff", "steps"]
WINDOWS = (3, 7, 14)
PHASES = ["menstruation", "late_follicular", "ovulation", "luteal"]


# --- causal primitives, matching infradian.features.causal exactly ---------------------

def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the last w values, min_periods=1, NaN-skipping (pandas rolling.mean)."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        win = x[max(0, i - w + 1) : i + 1]
        win = win[np.isfinite(win)]
        if win.size:
            out[i] = win.mean()
    return out


def _roll_std(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing sample std (ddof=1), min_periods=2, NaN -> 0.0 (pandas rolling.std().fillna(0))."""
    out = np.zeros(len(x))
    for i in range(len(x)):
        win = x[max(0, i - w + 1) : i + 1]
        win = win[np.isfinite(win)]
        out[i] = win.std(ddof=1) if win.size >= 2 else 0.0
    return out


def _delta(x: np.ndarray, lag: int) -> np.ndarray:
    """x[i] - x[i-lag]; NaN where unavailable (pandas s - s.shift(lag))."""
    out = np.full(len(x), np.nan)
    if lag < len(x):
        out[lag:] = x[lag:] - x[:-lag]
    return out


def _expanding_z(x: np.ndarray, min_periods: int = 5) -> np.ndarray:
    """Causal personal-baseline z-score from expanding mean/std, NaN -> 0.0."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        win = x[: i + 1]
        win = win[np.isfinite(win)]
        if win.size >= min_periods:
            sd = win.std(ddof=1)
            if sd and np.isfinite(sd) and sd != 0.0:
                out[i] = (x[i] - win.mean()) / sd
    return np.nan_to_num(out, nan=0.0)


def _cusum_positive(x: np.ndarray, drift: float) -> np.ndarray:
    z = _expanding_z(x)
    out = np.zeros(len(z))
    acc = 0.0
    for i in range(len(z)):
        acc = max(0.0, acc + z[i] - drift)
        out[i] = acc
    return out


def _days_since_last_onset(menses: np.ndarray) -> np.ndarray:
    m = np.nan_to_num(menses, nan=0.0).astype(int)
    out = np.full(len(m), -1.0)
    last = None
    for i in range(len(m)):
        onset = m[i] == 1 and (i == 0 or m[i - 1] == 0)
        if last is not None:
            out[i] = i - last
        if onset:
            last = i
            out[i] = 0.0
    return out


def _past_cycle_stats(menses: np.ndarray) -> dict[str, np.ndarray]:
    m = np.nan_to_num(menses, nan=0.0).astype(int)
    onsets = [i for i in range(len(m)) if m[i] == 1 and (i == 0 or m[i - 1] == 0)]
    median = np.full(len(m), np.nan)
    last = np.full(len(m), np.nan)
    count = np.zeros(len(m))
    completed: list[int] = []
    oi = 0
    for i in range(len(m)):
        while oi < len(onsets) and onsets[oi] <= i:
            if oi >= 1:
                completed.append(onsets[oi] - onsets[oi - 1])
            oi += 1
        if completed:
            median[i] = float(np.median(completed))
            last[i] = float(completed[-1])
            count[i] = len(completed)
    return {
        "cyclelen_median_past": median,
        "cyclelen_last_past": last,
        "cyclelen_count_past": count,
    }


# --- feature assembly -----------------------------------------------------------------

def build_feature_matrix(days: list[dict], feature_columns: list[str]) -> np.ndarray:
    """Build the (n_days, n_features) matrix in the exact column order the model was trained on."""
    n = len(days)
    col = lambda k: np.array(  # noqa: E731
        [float(d[k]) if d.get(k) is not None else np.nan for d in days], dtype=float
    )

    feats: dict[str, np.ndarray] = {}
    for sig in WEARABLE_SIGNALS:
        s = col(sig)
        feats[f"{sig}__z"] = _expanding_z(s)
        for w in WINDOWS:
            feats[f"{sig}__mean{w}"] = _roll_mean(s, w)
            feats[f"{sig}__std{w}"] = _roll_std(s, w)
        feats[f"{sig}__delta1"] = _delta(s, 1)
        feats[f"{sig}__delta7"] = _delta(s, 7)

    feats["skin_temp__cusum"] = _cusum_positive(col("skin_temp_dev_c"), drift=0.2)
    feats["rhr__cusum"] = _cusum_positive(col("rhr_bpm"), drift=0.2)

    menses = col("menses_reported")
    dsl = _days_since_last_onset(menses)
    feats["days_since_last_onset"] = dsl
    stats = _past_cycle_stats(menses)
    feats.update(stats)

    expected = np.where(np.isfinite(stats["cyclelen_median_past"]), stats["cyclelen_median_past"], 29.0)
    frac = np.clip(np.clip(dsl, 0, None) / expected, 0, 1.5)
    feats["cycle_frac_est"] = frac
    feats["cycle_frac_sin"] = np.sin(2 * np.pi * frac)
    feats["cycle_frac_cos"] = np.cos(2 * np.pi * frac)

    missing = [c for c in feature_columns if c not in feats]
    if missing:
        raise RuntimeError(f"numpy feature builder is missing columns: {missing}")

    return np.column_stack([feats[c] for c in feature_columns]).astype(float).reshape(n, -1)


@lru_cache(maxsize=1)
def load_models():
    """Load the models via the numpy booster reader plus the frozen feature-column order.

    Deliberately does not import lightgbm: it hard-imports scipy, which alone exceeds the
    serverless size budget. `_booster` reads the same model files and is verified bit-identical.
    """
    from _booster import load_model

    meta = json.loads((MODEL_DIR / "booster_meta.json").read_text())
    return {
        "meta": meta,
        "phase": load_model(str(MODEL_DIR / "phase_clf.txt")),
        "pdg": load_model(str(MODEL_DIR / "pdg_reg.txt")),
        "e3g": load_model(str(MODEL_DIR / "e3g_reg.txt")),
    }


def predict(days: list[dict]) -> dict:
    from _booster import predict as booster_predict

    m = load_models()
    X = build_feature_matrix(days, m["meta"]["feature_columns"])

    pdg = np.expm1(booster_predict(m["pdg"], X))
    e3g = np.expm1(booster_predict(m["e3g"], X))
    prob = np.asarray(booster_predict(m["phase"], X))
    if prob.ndim == 1:  # single-class edge case
        prob = prob.reshape(-1, 1)

    ov_class = PHASES.index("ovulation")
    classes = m["meta"]["phase_classes"]
    ov = prob[:, classes.index(ov_class)] if ov_class in classes else np.zeros(len(X))

    return {
        "days": [int(d["day_in_study"]) for d in days],
        "pdg_pred": [round(float(v), 3) for v in pdg],
        "e3g_pred": [round(float(v), 3) for v in e3g],
        "ovulation_prob": [round(float(v), 4) for v in ov],
    }
