"""Curated evidence store. The LLM may cite ONLY these claims, each traceable to a primary source
in docs/effect_sizes.md. Small enough that full-context injection beats a vector DB — so there is
no DB. Every citation the explanation renders resolves to one of these entries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    id: str
    claim: str
    source: str
    verification: str  # "verified" | "ranged"


EVIDENCE: list[Evidence] = [
    Evidence(
        "temp_luteal",
        "Nightly skin temperature rises about 0.20–0.50 °C from the follicular to the luteal phase.",
        "Grant et al., Int J Womens Health 2022 (Oura, n=26); wrist-vs-BBT fertility study.",
        "ranged",
    ),
    Evidence(
        "rhr_luteal",
        "Resting heart rate rises about 2–4 bpm from the follicular to the luteal phase.",
        "Grant et al. 2022 (+2.4 bpm); Shilaih et al., Sci Rep 2017 (+3.8 bpm); WHOOP (+2.7 bpm).",
        "verified",
    ),
    Evidence(
        "hrv_luteal",
        "Vagal HRV (RMSSD) decreases modestly from the follicular to the luteal phase.",
        "Grant et al. 2022 (−5.5 ms); Schmalenberger et al. meta-analysis (d≈−0.39).",
        "verified",
    ),
    Evidence(
        "temp_lags_ovulation",
        "The temperature rise is a consequence of post-ovulatory progesterone, appearing 1–3 days AFTER ovulation, so a wearable cannot forecast ovulation ahead of it.",
        "Established reproductive physiology (corpus luteum thermogenesis).",
        "verified",
    ),
    Evidence(
        "pdg_confirms_ovulation",
        "A sustained rise in urinary pregnanediol glucuronide (PdG) confirms that ovulation occurred.",
        "Ecochard-lineage urinary progesterone-metabolite criteria.",
        "verified",
    ),
    Evidence(
        "calendar_fails_irregular",
        "Calendar-based prediction assumes a fixed cycle length and fails most for irregular cycles.",
        "INFRADIAN benchmark, T2-P baseline: prospective calendar MAE ~5–8 days for irregular cycles.",
        "verified",
    ),
    Evidence(
        "cohort_scope",
        "The clinical cohort is 42 Canadian adults aged 18–29; results say nothing about perimenopause or PCOS-typical populations.",
        "mcPHASES data descriptor, Scientific Data 2026.",
        "verified",
    ),
    Evidence(
        "not_diagnostic",
        "These estimates are physiological, non-diagnostic, and not contraceptive or fertility guidance.",
        "INFRADIAN intended-use statement.",
        "verified",
    ),
]

EVIDENCE_BY_ID = {e.id: e for e in EVIDENCE}


def as_context() -> str:
    """Render the evidence store for full-context injection into the system prompt."""
    return "\n".join(f"[{e.id}] {e.claim} (Source: {e.source})" for e in EVIDENCE)
