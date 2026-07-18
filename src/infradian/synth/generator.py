"""INFRADIAN-SYNTH — a phenomenological synthetic cohort of menstrual cycles with coupled
consumer-wearable signals.

This is NOT a biophysical ODE model (no Reinecke-Deuflhard hypothalamic-pituitary-ovarian
system). It is a phenomenological generator: parametric hormone curves whose shapes match the
canonical endocrine literature, coupled to wearable channels through effect-size *ranges* that
are verified in docs/effect_sizes.md. The point is a CC-BY dataset anyone can run the whole
benchmark on with zero data-access friction, plus a controllable ground truth for validating the
harness.

Three physiological facts are built in on purpose, because they determine what the benchmark can
and cannot show:

1. Temperature/HR/HRV shifts are driven by PROGESTERONE (PdG), which rises only AFTER ovulation.
   So the wearable is a *lagged* proxy: it can reconstruct the luteal phase and confirm ovulation
   retrospectively, but it literally cannot forecast ovulation more than ~1 day ahead. Prospective
   skill at negative lead should be ~0 by construction — that is the honest, expected result.

2. A hydration nuisance term multiplies the (uncorrected) urinary hormone observations and is
   itself predictable from steps/temperature. This reproduces the real Mira dilution artifact and
   is exactly what the T1-NC follicular-only negative control is designed to bound.

3. Anovulatory cycles (concentrated in the irregular stratum) have no LH surge and no PdG rise, so
   they carry no luteal wearable signature. A calendar baseline assumes ovulation every cycle and
   therefore cannot detect these at all — the basis of task T2-A.

Determinism: fully seeded. No wall-clock, no global RNG.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from infradian.data import canonical as C

# Verified effect-size ranges (docs/effect_sizes.md). Luteal elevation vs personal follicular
# baseline, applied proportionally to normalized progesterone.
TEMP_LUTEAL_C = (0.20, 0.50)  # skin temperature deviation, C
RHR_LUTEAL_BPM = (2.0, 4.0)  # resting heart rate, bpm
HRV_LUTEAL_MS = (-8.0, -3.0)  # RMSSD, ms (a decrease)
RESP_LUTEAL_BR = (0.1, 0.5)  # respiratory rate, breaths/min

# Personal baselines (population means; each participant is offset from these).
RHR_BASE = (58.0, 72.0)
HRV_BASE = (35.0, 55.0)
RESP_BASE = (13.0, 17.0)
SLEEP_EFF_BASE = (0.82, 0.94)
STEPS_BASE = (6000.0, 12000.0)

ANOVULATORY_RATE_IRREGULAR = 0.18
ANOVULATORY_RATE_REGULAR = 0.03
IRREGULAR_FRACTION = 0.33

MISSING_BURST_RATE = (0.10, 0.25)  # fraction of days lost to off-wrist bursts
HYDRATION_ALPHA = 0.18  # strength of the (bounded) wearable->urine-concentration nuisance path


@dataclass
class ParticipantParams:
    """Latent per-participant parameters. Fixed for the participant across all their cycles."""

    pid: str
    regular: bool
    cycle_len_mu: float
    cycle_len_sigma: float
    luteal_len_mu: float
    menses_len_mu: float
    rhr_base: float
    hrv_base: float
    resp_base: float
    sleep_base: float
    steps_base: float
    temp_amp: float
    rhr_amp: float
    hrv_amp: float
    resp_amp: float
    anov_rate: float
    missing_rate: float
    # per-participant hormone amplitude multipliers (inter-individual variation)
    e3g_scale: float = 1.0
    pdg_scale: float = 1.0
    lh_scale: float = 1.0
    cycles: list[dict] = field(default_factory=list)


def _sample_participant(rng: np.random.Generator, pid: str) -> ParticipantParams:
    regular = rng.random() > IRREGULAR_FRACTION
    if regular:
        cycle_len_mu = rng.uniform(27.0, 31.0)
        cycle_len_sigma = rng.uniform(1.0, 2.5)
        anov = ANOVULATORY_RATE_REGULAR
    else:
        cycle_len_mu = rng.uniform(26.0, 40.0)
        cycle_len_sigma = rng.uniform(5.0, 9.0)  # the defining feature of the irregular stratum
        anov = ANOVULATORY_RATE_IRREGULAR

    return ParticipantParams(
        pid=pid,
        regular=regular,
        cycle_len_mu=cycle_len_mu,
        cycle_len_sigma=cycle_len_sigma,
        luteal_len_mu=rng.uniform(11.0, 14.0),  # luteal phase is relatively conserved
        menses_len_mu=rng.uniform(4.0, 6.5),
        rhr_base=rng.uniform(*RHR_BASE),
        hrv_base=rng.uniform(*HRV_BASE),
        resp_base=rng.uniform(*RESP_BASE),
        sleep_base=rng.uniform(*SLEEP_EFF_BASE),
        steps_base=rng.uniform(*STEPS_BASE),
        temp_amp=rng.uniform(*TEMP_LUTEAL_C),
        rhr_amp=rng.uniform(*RHR_LUTEAL_BPM),
        hrv_amp=rng.uniform(*HRV_LUTEAL_MS),
        resp_amp=rng.uniform(*RESP_LUTEAL_BR),
        anov_rate=anov,
        missing_rate=rng.uniform(*MISSING_BURST_RATE),
        e3g_scale=rng.uniform(0.7, 1.4),
        pdg_scale=rng.uniform(0.7, 1.4),
        lh_scale=rng.uniform(0.7, 1.4),
    )


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _hormone_curves(
    length: int, ov: int, anovulatory: bool, p: ParticipantParams, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Daily E3G, PdG, LH curves (arbitrary Mira-like units) over one cycle of given length.

    E3G: follicular rise peaking ~1d before ovulation, dip at ovulation, secondary luteal hump.
    LH:  sharp surge at ovulation (absent if anovulatory).
    PdG: low follicular, sigmoid rise ~2d after ovulation, luteal plateau, fall before menses
         (absent if anovulatory).
    """
    d = np.arange(length, dtype=float)

    zero = np.zeros(length)

    # E3G — present even in many anovulatory cycles (follicular estrogen still develops).
    e3g = (
        0.15
        + 0.9 * _gauss(d, ov - 1, 3.0)
        + (zero if anovulatory else 0.45 * _gauss(d, ov + 7, 4.0))
    ) * p.e3g_scale

    # LH — the surge. Anovulatory cycles have only baseline noise.
    lh = 0.10 + (zero if anovulatory else _gauss(d, ov, 0.8)) * p.lh_scale

    # PdG — luteal plateau. Rise starts ~2d post-ovulation, decays before next menses.
    if anovulatory:
        pdg = np.full(length, 0.08) * p.pdg_scale
    else:
        rise = _sigmoid((d - (ov + 2)) / 1.3)
        fall = 1.0 - _sigmoid((d - (length - 2)) / 1.3)
        pdg = (0.08 + 0.95 * rise * fall) * p.pdg_scale

    # Biological day-to-day noise (multiplicative, small).
    for arr in (e3g, lh, pdg):
        arr *= rng.normal(1.0, 0.06, size=length).clip(0.6, 1.5)

    return {"e3g": np.maximum(e3g, 0.01), "pdg": np.maximum(pdg, 0.01), "lh": np.maximum(lh, 0.01)}


def _phase_labels(length: int, ov: int, menses_len: int, anovulatory: bool) -> np.ndarray:
    """Four-phase labels following the mcPHASES operational definitions.

    Anovulatory cycles are still labelled by the same day geometry (the phase labeller in the real
    data is proprietary and would also assign phases); the absence of a hormone signature is what
    task T2-A detects, not a missing label.
    """
    labels = np.empty(length, dtype=object)
    fertile_start = max(menses_len, ov - 5)
    for i in range(length):
        if i < menses_len:
            labels[i] = "menstruation"
        elif i < fertile_start:
            labels[i] = "late_follicular"
        elif i <= ov:
            labels[i] = "ovulation"
        else:
            labels[i] = "luteal"
    return labels


def _generate_participant(p: ParticipantParams, n_days: int, rng: np.random.Generator) -> pd.DataFrame:
    """Assemble one participant's full time series in canonical long format."""
    rows: list[dict] = []
    day = 0
    while day < n_days:
        length = int(round(rng.normal(p.cycle_len_mu, p.cycle_len_sigma)))
        length = int(np.clip(length, 20, 60))
        menses_len = int(round(rng.normal(p.menses_len_mu, 0.8)))
        menses_len = int(np.clip(menses_len, 3, 8))
        luteal_len = int(round(rng.normal(p.luteal_len_mu, 1.2)))
        luteal_len = int(np.clip(luteal_len, 10, 16))
        ov = max(menses_len + 3, length - luteal_len)  # follicular varies; luteal ~fixed
        anovulatory = rng.random() < p.anov_rate

        curves = _hormone_curves(length, ov, anovulatory, p, rng)
        phases = _phase_labels(length, ov, menses_len, anovulatory)

        # Normalized progesterone (0..1) drives the lagged wearable elevation. We rescale so its
        # LUTEAL-PHASE MEAN is ~1.0, because the effect-size literature reports luteal-vs-follicular
        # phase *means*, not peaks. With this scaling, `amp * luteal_drive` yields an achieved
        # phase-mean delta equal to the sampled amplitude, so the generator matches docs/effect_sizes.md.
        pdg_norm = (curves["pdg"] - curves["pdg"].min()) / (np.ptp(curves["pdg"]) + 1e-9)
        luteal_mask = phases == "luteal"
        luteal_mean = pdg_norm[luteal_mask].mean() if luteal_mask.any() else 1.0
        luteal_drive = pdg_norm / (luteal_mean + 1e-9)

        # Nuisance driver days (illness / alcohol / travel): independent bumps to temp/RHR/HRV.
        nuisance = (rng.random(length) < 0.05).astype(float)
        nuisance_temp = nuisance * rng.normal(0.25, 0.1, length)
        nuisance_rhr = nuisance * rng.normal(4.0, 1.5, length)

        cd = np.arange(length)
        for i in range(length):
            if day + i >= n_days:
                break
            # Wearable channels: personal baseline + lagged luteal (progesterone) elevation + noise.
            skin_temp = p.temp_amp * luteal_drive[i] + nuisance_temp[i] + rng.normal(0, 0.06)
            rhr = p.rhr_base + p.rhr_amp * luteal_drive[i] + nuisance_rhr[i] + rng.normal(0, 1.2)
            hrv = p.hrv_base + p.hrv_amp * luteal_drive[i] + rng.normal(0, 3.0)
            resp = p.resp_base + p.resp_amp * luteal_drive[i] + rng.normal(0, 0.4)
            steps = max(0.0, p.steps_base * rng.normal(1.0, 0.25))
            sleep_eff = float(np.clip(p.sleep_base + rng.normal(0, 0.04), 0.5, 1.0))

            # Hydration nuisance: urinary metabolite concentration scales with a factor that is
            # itself predictable from steps and temperature. This is the dilution artifact the
            # T1-NC control bounds. Bounded so it never dominates the true hormone signal.
            z_steps = (steps - p.steps_base) / (0.25 * p.steps_base)
            hydration = np.exp(HYDRATION_ALPHA * (0.6 * z_steps + 0.4 * skin_temp / 0.3))
            hydration = float(np.clip(hydration, 0.7, 1.4))

            rows.append(
                {
                    C.KEY_PARTICIPANT: p.pid,
                    C.KEY_SEGMENT: C.make_segment_id(p.pid, 0),
                    C.KEY_DAY: day + i,
                    "skin_temp_dev_c": round(skin_temp, 4),
                    "rhr_bpm": round(rhr, 2),
                    "hrv_rmssd_ms": round(hrv, 2),
                    "resp_rate": round(resp, 3),
                    "sleep_eff": round(sleep_eff, 4),
                    "steps": int(steps),
                    "menses_reported": 1 if cd[i] < menses_len else 0,
                    "cramps": int(np.clip(rng.poisson(2.0 if cd[i] < menses_len else 0.4), 0, 5)),
                    "mood": int(np.clip(rng.normal(3 - 0.8 * (cd[i] > ov), 1.0), 0, 5)),
                    "stress": int(np.clip(rng.normal(2.5, 1.0), 0, 5)),
                    "e3g": round(float(curves["e3g"][i] * hydration), 4),
                    "pdg": round(float(curves["pdg"][i] * hydration), 4),
                    "lh": round(float(curves["lh"][i] * hydration), 4),
                    "phase": phases[i],
                    # metadata columns (dropped before publishing features; used for scoring/labels)
                    "_ovulation_day": -1 if anovulatory else (day + ov),
                    "_anovulatory": int(anovulatory),
                    "_cycle_start_day": day,
                }
            )
        day += length

    df = pd.DataFrame(rows)

    # Apply burst missingness to wearable channels only (hormones/diary stay — Mira/diary are
    # daily-adherence, wearable gaps are off-wrist).
    _apply_missing_bursts(df, p.missing_rate, rng)
    return df


def _apply_missing_bursts(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> None:
    """Knock out contiguous bursts of wearable days to reach approximately `rate` missing."""
    n = len(df)
    target = int(rate * n)
    lost = 0
    guard = 0
    while lost < target and guard < 1000:
        guard += 1
        start = rng.integers(0, n)
        burst = rng.integers(1, 5)
        idx = df.index[start : start + burst]
        df.loc[idx, C.WEARABLE_COLUMNS] = np.nan
        lost += len(idx)


def generate_cohort(n: int = 600, n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    """Generate the full synthetic cohort in canonical long format (plus metadata columns)."""
    rng = np.random.default_rng(seed)
    frames = []
    width = len(str(n))
    for k in range(n):
        pid = f"S{k:0{width}d}"
        params = _sample_participant(rng, pid)
        frames.append(_generate_participant(params, n_days, rng))
    df = pd.concat(frames, ignore_index=True)
    return df


def assign_splits(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Participant-disjoint train/validation/test assignment (70/15/15), stratified by regularity.

    The HF dataset ships these splits; the benchmark's own CV happens within train.
    """
    rng = np.random.default_rng(seed + 1)
    pids = df[C.KEY_PARTICIPANT].unique()
    # infer regularity from cycle-length spread per participant
    reg_map = {}
    for pid, g in df.groupby(C.KEY_PARTICIPANT):
        # reconstruct cycle lengths from menses onsets
        onsets = g.loc[g["menses_reported"] == 1, C.KEY_DAY].to_numpy()
        starts = onsets[np.concatenate([[True], np.diff(onsets) > 1])]
        lens = np.diff(starts)
        irregular = (len(lens) > 0) and (
            (np.ptp(lens) >= 9) or bool(((lens < 24) | (lens > 38)).any())
        )
        reg_map[pid] = "irregular" if irregular else "regular"

    split = {}
    for stratum in ("regular", "irregular"):
        members = [pid for pid in pids if reg_map[pid] == stratum]
        rng.shuffle(members)
        n_test = max(1, int(0.15 * len(members)))
        n_val = max(1, int(0.15 * len(members)))
        for i, pid in enumerate(members):
            if i < n_test:
                split[pid] = "test"
            elif i < n_test + n_val:
                split[pid] = "validation"
            else:
                split[pid] = "train"

    df = df.copy()
    df["split"] = df[C.KEY_PARTICIPANT].map(split)
    df["regularity"] = df[C.KEY_PARTICIPANT].map(reg_map)
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate the INFRADIAN-SYNTH cohort.")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/tier_c"))
    args = ap.parse_args(argv)

    df = generate_cohort(args.n, args.days, args.seed)
    df = assign_splits(df, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    # Full frame (with metadata) for internal use.
    df.to_parquet(args.out / "cohort.parquet", index=False)

    # Publishable splits: drop leakage-prone metadata columns for the HF release.
    public_cols = [c for c in df.columns if not c.startswith("_")]
    for split_name in ("train", "validation", "test"):
        sub = df.loc[df["split"] == split_name, public_cols]
        sub.to_parquet(args.out / f"{split_name}-00000-of-00001.parquet", index=False)

    n_participants = df[C.KEY_PARTICIPANT].nunique()
    n_irregular = df.loc[df["regularity"] == "irregular", C.KEY_PARTICIPANT].nunique()
    anov_cycles = df.loc[df["_anovulatory"] == 1].groupby(C.KEY_PARTICIPANT).ngroups
    print(f"generated {n_participants} participants, {len(df)} participant-days")
    print(f"  irregular: {n_irregular} ({n_irregular / n_participants:.0%})")
    print(f"  participants with >=1 anovulatory cycle: {anov_cycles}")
    print(f"  wrote {args.out}/cohort.parquet + train/validation/test splits")


if __name__ == "__main__":
    main()
