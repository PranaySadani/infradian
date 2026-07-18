"""Assert the synthetic generator reproduces the verified physiological effect sizes.

IMPORTANT CAVEAT (stated so nobody over-reads this test): this validates the generator against
ITS OWN INPUTS — the ranges in docs/effect_sizes.md — not against nature. It catches regressions
and units bugs (e.g. an RHR delta drifting to fever-magnitude, or a sign flip), which is exactly
the failure mode that would otherwise silently falsify the dataset card and the LLM evidence store.
"""

from __future__ import annotations

import numpy as np
import pytest

from infradian.synth.generator import generate_cohort

# Target ranges (luteal minus follicular phase means) from docs/effect_sizes.md.
TARGETS = {
    "skin_temp_dev_c": (0.18, 0.55),
    "rhr_bpm": (1.8, 4.2),
    "hrv_rmssd_ms": (-9.0, -2.5),
    "resp_rate": (0.08, 0.55),
}


@pytest.fixture(scope="module")
def cohort():
    return generate_cohort(n=120, n_days=120, seed=7)


def _luteal_minus_follicular(df, col: str) -> float:
    deltas = []
    for _, g in df.groupby("participant_id"):
        fol = g[g.phase.isin(["menstruation", "late_follicular"])][col].mean()
        lut = g[g.phase == "luteal"][col].mean()
        if np.isfinite(fol) and np.isfinite(lut):
            deltas.append(lut - fol)
    return float(np.nanmean(deltas))


@pytest.mark.parametrize("col", list(TARGETS))
def test_effect_size_in_range(cohort, col):
    lo, hi = TARGETS[col]
    delta = _luteal_minus_follicular(cohort, col)
    assert lo <= delta <= hi, f"{col} luteal-follicular delta {delta:+.3f} outside [{lo}, {hi}]"


def test_lh_peaks_at_ovulation(cohort):
    ov = cohort[cohort._anovulatory == 0]
    means = ov.groupby("phase")["lh"].mean()
    assert means.idxmax() == "ovulation"


def test_pdg_peaks_in_luteal(cohort):
    ov = cohort[cohort._anovulatory == 0]
    means = ov.groupby("phase")["pdg"].mean()
    assert means.idxmax() == "luteal"


def test_anovulatory_cycles_have_no_lh_surge(cohort):
    anov = cohort[cohort._anovulatory == 1]
    ovul = cohort[cohort._anovulatory == 0]
    # Anovulatory ovulation-phase LH should be far below ovulatory ovulation-phase LH.
    anov_lh = anov[anov.phase == "ovulation"]["lh"].mean()
    ovul_lh = ovul[ovul.phase == "ovulation"]["lh"].mean()
    assert anov_lh < 0.5 * ovul_lh


def test_irregular_stratum_exists_and_is_minority(cohort):
    from infradian.synth.generator import assign_splits

    df = assign_splits(cohort, seed=7)
    frac = df.groupby("participant_id")["regularity"].first().eq("irregular").mean()
    assert 0.15 <= frac <= 0.55, f"irregular fraction {frac:.2f} implausible"


def test_missingness_is_present_but_bounded(cohort):
    nan_frac = cohort["skin_temp_dev_c"].isna().mean()
    assert 0.05 <= nan_frac <= 0.35
    # Hormones should never be missing in synthetic (Mira daily adherence assumed).
    assert cohort["pdg"].notna().all()
