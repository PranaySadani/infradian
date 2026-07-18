# Limitations

Written plainly, because a benchmark that hides its limits is not a benchmark.

## Sample and population
- The clinical cohort (mcPHASES) is **42 Canadian adults, ages 18–29**, no hormonal contraception, no
  diabetes, 192 cycles. It says **nothing** about perimenopause, PCOS-typical populations, older adults,
  or more ethnically diverse groups. Do not generalize any number here to those populations.
- With n=42, confidence intervals on stratified real-data metrics are wide. We report them and do not
  narrate differences the intervals do not support.

## What the result is
- On real consumer wearables the **primary endpoint is null** (SoC −0.07, p=0.48). Wearable-derived
  ovulation-day inference does not beat a strong calendar baseline here. Cycle-phase information is
  present (macro-F1 0.46 vs 0.25 chance), but day-level hormone reconstruction and ovulation timing are
  not solved by today's consumer wearables. This is a finding, not a failure — but it is a *negative*
  finding, and should be cited as such.

## Known confounds
- **Season / ambient temperature.** Skin temperature is the strongest signal, and Canadian ambient
  temperature across a 3-month window can dwarf a ~0.3 °C luteal shift. mcPHASES **stripped absolute
  dates**, so seasonal drift cannot be adjusted for or even measured. Short-window features (delta-7,
  CUSUM) are largely immune; long-window baseline-deviation features are exposed. We ablate for it.
- **Urine dilution.** Mira urinary hormones are not creatinine-normalized. The T1-NC follicular-only
  negative control bounds this artifact (real data: ρ≈0.02 — negligible; synthetic: ρ≈0.30 by design).
- **Proprietary phase labels.** mcPHASES phase labels come from Mira's proprietary algorithm. We build
  headline claims on measured hormone values and LH-derived ovulation, and treat the 4-phase labels as a
  secondary, clearly-caveated task.

## Method caveats
- The synthetic cohort is **phenomenological**, not a biophysical ODE model. It is calibrated to *match*
  published effect sizes; `test_synthetic_effect_sizes.py` validates it against its own inputs, not
  against nature.
- Ovulation ground truth on real data is anchored on the LH surge (PdG is sampled on only ~35% of days).
  The reported "cycles without a detected PdG rise" is a measurement statement, not a clinical
  anovulation-prevalence claim.
- The reference checkpoint is trained on synthetic data only; its real-data transfer is weak by the same
  sim-to-real gap the benchmark measures.

## Intended use
Non-diagnostic research infrastructure. **Not** a medical device, **not** contraceptive or fertility
guidance. The LLM explanation layer refuses diagnostic, treatment, and contraception questions.
