"""The canonical daily schema — FROZEN. Every module in the package keys off these
column names: the synthetic generator, the feature builder, the model bundle, the
API schema, the exported TypeScript types, and every figure.

Renaming a column here is a breaking change that ripples through the whole system,
so it is deliberately small and stable.

Two design decisions that are load-bearing (see plan §7.1):

1. All wearable channels are RELATIVE or z-scored. Fitbit Sense reports nightly
   *relative* skin-temperature deviation (~ -2..+2 C around a personal baseline),
   NOT absolute skin temperature; and its HRV is RMSSD-family (~20-60 ms), NOT SDNN.
   There is deliberately no absolute-temperature field. Calibrating a generator to an
   SDNN effect size and testing on Fitbit RMSSD would collapse sim-to-real transfer
   for units reasons and look like distribution shift.

2. Two keys, never conflated. `participant_id` is the SPLITTING key (grouping unit for
   participant-disjoint folds). `segment_id` is the TEMPORAL key: it encodes the study
   round so rolling windows and cycle detection reset at the round boundary (mcPHASES
   Interval 2 timelines resume ~day 905, a two-year real-time gap).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SCHEMA_VERSION = "1.0.0"

# --- keys ---
KEY_PARTICIPANT = "participant_id"  # splitting key — NEVER includes round
KEY_SEGMENT = "segment_id"  # temporal key — f"{participant_id}__r{round_idx}"
KEY_DAY = "day_in_study"  # integer day; restarts per round

# --- wearable channels (all relative / z-scored) ---
WEARABLE_COLUMNS = [
    "skin_temp_dev_c",  # nightly relative skin-temp deviation (C), personal baseline = 0
    "rhr_bpm",  # resting heart rate (bpm) — absolute, but z-scored per participant in features
    "hrv_rmssd_ms",  # nightly HRV, RMSSD family (ms)
    "resp_rate",  # respiratory rate (breaths/min)
    "sleep_eff",  # sleep efficiency (0..1)
    "steps",  # daily step count
]

# --- self-report / diary ---
DIARY_COLUMNS = [
    "menses_reported",  # 0/1 self-reported bleeding that day (defines the menstruation label)
    "cramps",  # 0..5 self-reported cramp severity
    "mood",  # 0..5 self-reported mood
    "stress",  # 0..5 self-reported stress
]

# --- hormone ground truth (Mira urinary, uncorrected for hydration) ---
HORMONE_COLUMNS = [
    "e3g",  # estrone-3-glucuronide (estradiol metabolite)
    "pdg",  # pregnanediol glucuronide (progesterone metabolite)
    "lh",  # luteinizing hormone
]

# --- labels ---
LABEL_COLUMNS = [
    "phase",  # {menstruation, late_follicular, ovulation, luteal} — proprietary-derived, caveated
]

PHASES = ["menstruation", "late_follicular", "ovulation", "luteal"]

ALL_COLUMNS = (
    [KEY_PARTICIPANT, KEY_SEGMENT, KEY_DAY]
    + WEARABLE_COLUMNS
    + DIARY_COLUMNS
    + HORMONE_COLUMNS
    + LABEL_COLUMNS
)

# Feature registry denylist: names that leak the label. The feature builder raises
# if any of these appear as a feature (plan §8.4 L1, L7).
BANNED_FEATURE_NAMES = frozenset(
    {
        "days_until_next_menses",
        "days_until_ovulation",
        "phase",
        "cycle_day",  # only days_since_last_OBSERVED_menses_onset is legal
    }
)


@dataclass(frozen=True)
class FeatureSpec:
    """Records exactly which columns are inputs vs targets, so train and serve agree.

    The spec is hashed and stored with every model bundle; the API loads the same spec
    to guarantee train/serve feature parity (plan §11).
    """

    wearable: tuple[str, ...] = tuple(WEARABLE_COLUMNS)
    diary: tuple[str, ...] = tuple(DIARY_COLUMNS)
    hormone_targets: tuple[str, ...] = ("pdg", "e3g")
    schema_version: str = SCHEMA_VERSION


def empty_frame() -> pd.DataFrame:
    """An empty dataframe with exactly the canonical columns and no rows."""
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in ALL_COLUMNS})


def validate(df: pd.DataFrame, *, require_hormones: bool = False) -> pd.DataFrame:
    """Assert a dataframe conforms to the canonical schema. Returns it unchanged.

    Raises ValueError on any structural violation so shape bugs fail loud and early.
    """
    missing = [c for c in [KEY_PARTICIPANT, KEY_SEGMENT, KEY_DAY] if c not in df.columns]
    if missing:
        raise ValueError(f"canonical frame missing key columns: {missing}")

    for c in WEARABLE_COLUMNS + DIARY_COLUMNS + LABEL_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"canonical frame missing column: {c}")

    if require_hormones:
        for c in HORMONE_COLUMNS:
            if c not in df.columns:
                raise ValueError(f"canonical frame missing hormone column: {c}")

    # segment_id must embed participant_id (invariant used by the split logic).
    bad = df[~df[KEY_SEGMENT].astype(str).str.startswith(df[KEY_PARTICIPANT].astype(str))]
    if len(bad) > 0:
        raise ValueError(
            f"{len(bad)} rows have segment_id not prefixed by participant_id "
            "(violates the splitting invariant)"
        )

    if "phase" in df.columns:
        bad_phase = set(df["phase"].dropna().unique()) - set(PHASES)
        if bad_phase:
            raise ValueError(f"unknown phase labels: {bad_phase}")

    return df


def make_segment_id(participant_id: str, round_idx: int) -> str:
    """The one place segment_id is constructed. Round is 0-based."""
    return f"{participant_id}__r{round_idx}"
