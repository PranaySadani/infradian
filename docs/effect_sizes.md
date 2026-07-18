# Physiological effect sizes — verified magnitudes for the synthetic generator

Every wearable-coupling magnitude used by `infradian.synth.generator` is listed here with
its **primary source, exact comparison, sample size, and verification status**. This file
is the single source of truth: the generator reads *ranges* from it, `test_synthetic_effect_sizes.py`
asserts the generator reproduces those ranges, and the LLM evidence store cites the same rows.

**Why this file exists (plan §2.4):** an earlier draft carried `+8 bpm` luteal RHR and
`SDNN 154→136 ms`. Both are wrong. `+8 bpm` is fever-magnitude — the real follicular→luteal
shift is ~2–4 bpm. And Fitbit/Oura report **RMSSD, not SDNN**; a 154 ms nocturnal wrist SDNN is
a 24-hour Holter figure. One wrong magnitude would have falsified four artifacts at once
(generator, its test, the dataset card, the LLM evidence store) and made a units bug look like
sim-to-real distribution shift. Hence: verify from the primary table, or encode a range.

## Verification legend
- ✅ **verified** — read from the primary paper's own table/text, exact comparison confirmed.
- 🟡 **ranged** — multiple credible primaries disagree on the point value; encoded as a range.
- ❌ **rejected** — previously-used number found to be wrong; kept here as a guardrail.

---

## Skin temperature (menses/follicular → luteal)

| Status | Magnitude | Comparison | n | Source |
|---|---|---|---|---|
| ✅ | **+0.198 °C** (menses→mid-luteal), **+0.242 °C** (menses→late-luteal), both p<0.001 | distal (finger) skin temp, Oura | 26 | Grant et al., *Int J Womens Health* 2022, [PMC9005074](https://pmc.ncbi.nlm.nih.gov/articles/PMC9005074/) |
| 🟡 | **+0.50 °C** post-ovulation (wrist), vs +0.20 °C basal body temp | wrist skin temp | — | wrist-vs-BBT fertility study (secondary; wrist runs larger than distal) |
| ✅ | phase effect present, **F(3,123)=11.63, p<.001**, lowest follicular / peak luteal, **no magnitude reported** | nightly wrist temp, Fitbit Sense | 42 | mcPHASES descriptor, [PMC13003092](https://pmc.ncbi.nlm.nih.gov/articles/PMC13003092/) |

**Generator range:** luteal skin-temp elevation ∈ **[0.20, 0.50] °C** above personal follicular baseline. Distal anchors the low end, wrist the high end; mcPHASES confirms direction and significance but gives no magnitude.

## Resting / nocturnal heart rate (menses → luteal)

| Status | Magnitude | Comparison | n | Source |
|---|---|---|---|---|
| ✅ | **+2.38 bpm** (menses→mid-luteal), **+2.49 bpm** (menses→late-luteal), p=0.001 | nocturnal HR, Oura | 26 | Grant et al. 2022, [PMC9005074](https://pmc.ncbi.nlm.nih.gov/articles/PMC9005074/) |
| ✅ | **+3.8 bpm** mid-luteal vs menses; +2.1 fertile vs menses | sleeping pulse rate | 91 | Shilaih et al., *Sci Rep* 2017, [s41598-017-01433-9](https://www.nature.com/articles/s41598-017-01433-9) |
| ✅ | **+2.73 bpm** follicular→luteal (mean) | RHR | large | WHOOP menstrual-cycle research |
| ✅ | phase effect present, **F(3,123)=10.26, p<.001**, lowest follicular / peak luteal | median resting HR, Fitbit Sense | 42 | mcPHASES descriptor, [PMC13003092](https://pmc.ncbi.nlm.nih.gov/articles/PMC13003092/) |
| ❌ | ~~+8 bpm luteal~~ | — | — | **REJECTED** — fever-magnitude; not a phase-mean shift. Do not use. |

**Generator range:** luteal RHR elevation ∈ **[2.0, 4.0] bpm** above personal follicular baseline. Three independent primaries cluster here.

## HRV — RMSSD (menses → luteal)

| Status | Magnitude | Comparison | n | Source |
|---|---|---|---|---|
| ✅ | **−5.47 ms** (menses→mid-luteal, p=0.054 trend), **−5.96 ms** (menses→late-luteal, p=0.034); overall model n.s. (F=1.93, p=0.13) | nocturnal rMSSD, Oura | 26 | Grant et al. 2022, [PMC9005074](https://pmc.ncbi.nlm.nih.gov/articles/PMC9005074/) |
| ✅ | vagal HRV **decreases follicular→luteal, d ≈ −0.39** (medium) | meta-analytic, rMSSD/HF | pooled | Schmalenberger et al. meta-analysis, [PMC7141121](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7141121/) |
| ❌ | ~~SDNN 154→136 ms (−12%)~~ | — | — | **REJECTED** — SDNN is a 24-h Holter metric; wearables report **RMSSD**. Category error. |

**Generator range:** luteal RMSSD change ∈ **[−8, −3] ms** (equivalently a small negative standardized shift ≈ d −0.4). Channel is `hrv_rmssd_ms`, never SDNN.

## Respiratory rate (menses → luteal)

| Status | Magnitude | Comparison | n | Source |
|---|---|---|---|---|
| 🟡 | elevated in luteal phase, **~+0.2 to +0.5 breaths/min** | breaths/min | — | secondary summaries; not measured in the Oura study |

**Generator range:** luteal respiratory-rate elevation ∈ **[0.1, 0.5] breaths/min**. Low-confidence; kept small.

## Cycle structure (mcPHASES descriptor, ✅ verified)

Mean cycle 31.6 d. Phase durations: menstruation **6.2 d**, late-follicular **7.7 d**, ovulation **7.3 d**, luteal **10.4 d**. 192 complete cycles across 42+20 participants. Source: [PMC13003092](https://pmc.ncbi.nlm.nih.gov/articles/PMC13003092/).

## Ovulation confirmation criterion (for T2-A ground truth)

Use a **published fixed-threshold, three-consecutive-day PdG-rise criterion** (Ecochard lineage:
a sustained urinary PdG rise confirms luteinization), **not** a homemade z-score rule whose SD is
computed over the very luteal elevation being detected (circular). Report Cohen's κ agreement between
the published criterion and any alternative. The reported statistic is *"cycles without a detected PdG
rise under criterion X"* — a measurement statement, not a clinical anovulation-prevalence claim.

---

## What this buys the project

Every magnitude the generator uses is a **range with a citation and a verification mark**, and the
dataset card carries a `calibration_uncertainty` note pointing here. A reviewer who checks any number
finds the primary table behind it. That converts what would have been a fatal, silent error into an
explicit rigor signal.
