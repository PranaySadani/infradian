"""mcPHASES loader → canonical schema. THE ONE SEAM between restricted data and the harness.

Everything downstream is built against the synthetic tier, which shares this exact schema; this
module is the only place the real data enters. Nothing here is ever committed or published — the
raw data is PhysioNet Restricted Health Data License 1.5.0. Only derived aggregate metrics leave.

Design notes discovered during the CP2 timebox:
  - Keys are `id` + `day_in_study`; `study_interval` (2022 / 2024) is the round marker, so the
    segment_id is built directly from it rather than by detecting the ~day-905 discontinuity.
  - Skin temperature is `temperature_diff_from_baseline` (RELATIVE) in wrist_temperature.csv —
    confirming the canonical schema's relative-channel decision.
  - No creatinine / specific-gravity column exists, so the urine-dilution artifact is bounded by
    the T1-NC follicular-only negative control rather than corrected.
  - The multi-GB intraday tables (heart_rate, calories, distance, steps) are NOT read; the daily
    summary tables carry everything the model needs.
  - PdG is measured on far fewer days than LH/E3G (it is the luteal progesterone metabolite),
    which is expected and handled — PdG regression simply has fewer labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from infradian.data import canonical as C

# Proprietary phase names -> canonical four-phase vocabulary.
PHASE_MAP = {
    "Menstrual": "menstruation",
    "Follicular": "late_follicular",
    "Fertility": "ovulation",
    "Luteal": "luteal",
}

# Ordinal self-report scale -> 0..4.
ORDINAL = {
    "Not at all": 0,
    "Very Low/Little": 0,
    "Low": 1,
    "Somewhat Light": 1,
    "Moderate": 2,
    "Somewhat Heavy": 3,
    "High": 3,
    "Very High": 4,
    "Heavy": 4,
}

# Flow values that count as actual bleeding (define the self-reported menstruation label).
_NO_FLOW = {"Not at all"}


def _round_idx(interval: int) -> int:
    return 0 if int(interval) == 2022 else 1


def _daily_mean(path: Path, value_col: str, out_col: str) -> pd.DataFrame:
    """Aggregate an intraday/summary table to one value per (id, interval, day)."""
    df = pd.read_csv(path, usecols=["id", "study_interval", "day_in_study", value_col])
    g = (
        df.groupby(["id", "study_interval", "day_in_study"])[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: out_col})
    )
    return g


def load_canonical(root: str | Path) -> pd.DataFrame:
    """Load mcPHASES daily tables and return a canonical long dataframe.

    Missing wearable channels are left as NaN (the feature builder and LightGBM tolerate NaN).
    """
    root = Path(root)
    hs = pd.read_csv(root / "hormones_and_selfreport.csv")

    base = pd.DataFrame(
        {
            C.KEY_PARTICIPANT: hs["id"].astype(str),
            "round": hs["study_interval"].map(_round_idx),
            C.KEY_DAY: hs["day_in_study"].astype(int),
            "e3g": pd.to_numeric(hs["estrogen"], errors="coerce"),
            "pdg": pd.to_numeric(hs["pdg"], errors="coerce"),
            "lh": pd.to_numeric(hs["lh"], errors="coerce"),
            "phase": hs["phase"].map(PHASE_MAP),
            "cramps": hs["cramps"].map(ORDINAL),
            "mood": hs["moodswing"].map(ORDINAL),
            "stress": hs["stress"].map(ORDINAL),
            "menses_reported": (~hs["flow_volume"].isin(_NO_FLOW) & hs["flow_volume"].notna()).astype(int),
            "_interval": hs["study_interval"].astype(int),
        }
    )
    base[C.KEY_SEGMENT] = [
        C.make_segment_id(pid, r) for pid, r in zip(base[C.KEY_PARTICIPANT], base["round"], strict=True)
    ]

    # Daily wearable aggregations (small/medium tables only).
    wearables = {
        "skin_temp_dev_c": ("wrist_temperature.csv", "temperature_diff_from_baseline"),
        "rhr_bpm": ("resting_heart_rate.csv", "value"),
        "hrv_rmssd_ms": ("heart_rate_variability_details.csv", "rmssd"),
        "resp_rate": ("respiratory_rate_summary.csv", "full_sleep_breathing_rate"),
    }
    for out_col, (fname, val_col) in wearables.items():
        path = root / fname
        if not path.exists():
            base[out_col] = np.nan
            continue
        agg = _daily_mean(path, val_col, out_col)
        agg[C.KEY_PARTICIPANT] = agg["id"].astype(str)
        agg[C.KEY_DAY] = agg["day_in_study"].astype(int)
        agg["round"] = agg["study_interval"].map(_round_idx)
        base = base.merge(
            agg[[C.KEY_PARTICIPANT, "round", C.KEY_DAY, out_col]],
            on=[C.KEY_PARTICIPANT, "round", C.KEY_DAY],
            how="left",
        )

    # sleep efficiency proxy from sleep_score (overall_score / 100).
    ss_path = root / "sleep_score.csv"
    if ss_path.exists():
        ss = pd.read_csv(ss_path, usecols=["id", "study_interval", "day_in_study", "overall_score"])
        ss = (
            ss.groupby(["id", "study_interval", "day_in_study"])["overall_score"].mean().reset_index()
        )
        ss[C.KEY_PARTICIPANT] = ss["id"].astype(str)
        ss[C.KEY_DAY] = ss["day_in_study"].astype(int)
        ss["round"] = ss["study_interval"].map(_round_idx)
        ss["sleep_eff"] = ss["overall_score"] / 100.0
        base = base.merge(
            ss[[C.KEY_PARTICIPANT, "round", C.KEY_DAY, "sleep_eff"]],
            on=[C.KEY_PARTICIPANT, "round", C.KEY_DAY],
            how="left",
        )
    else:
        base["sleep_eff"] = np.nan

    base["steps"] = np.nan  # intraday steps.csv (227 MB) intentionally not read

    # Order columns to the canonical layout; drop helper columns.
    for col in C.ALL_COLUMNS:
        if col not in base.columns:
            base[col] = np.nan
    out = base[C.ALL_COLUMNS].copy()
    out = out.sort_values([C.KEY_SEGMENT, C.KEY_DAY]).reset_index(drop=True)
    C.validate(out)
    return out


def coverage_report(df: pd.DataFrame) -> dict:
    """Summary statistics used for the go/no-go decision and the dataset card aggregates."""
    wearable_present = df[C.WEARABLE_COLUMNS].notna().sum(axis=1)
    hormone_present = df[["lh", "e3g", "pdg"]].notna().any(axis=1)
    return {
        "n_participants": int(df[C.KEY_PARTICIPANT].nunique()),
        "n_segments": int(df[C.KEY_SEGMENT].nunique()),
        "n_participant_days": int(len(df)),
        "days_with_hormone": int(hormone_present.sum()),
        "days_with_hormone_and_3plus_wearables": int((hormone_present & (wearable_present >= 3)).sum()),
        "pdg_days": int(df["pdg"].notna().sum()),
        "lh_days": int(df["lh"].notna().sum()),
        "e3g_days": int(df["e3g"].notna().sum()),
        "wearable_null_frac": {
            c: round(float(df[c].isna().mean()), 3) for c in C.WEARABLE_COLUMNS
        },
    }
