# INFRADIAN-BENCH v1.0 — specification

A frozen, citable specification of the benchmark. Re-implementing it should reproduce our numbers.

## Prediction-time contract
Every task is evaluated **at end of day `t`, using only data timestamped ≤ end of day `t`.** Enforced
by `tests/test_feature_causality.py` (future-perturbation + truncation-invariance over all features).

## Canonical schema
One row per `(participant_id, day_in_study)`. Wearable channels are **relative / z-scored** (Fitbit
reports relative skin-temperature deviation and RMSSD, never absolute temperature or SDNN). Two keys,
never conflated: `participant_id` (splitting) and `segment_id = participant_id__r{round}` (temporal —
resets rolling windows at the study-round boundary). See `src/infradian/data/canonical.py`.

## Tasks

| ID | Task | Target | Primary metric |
|---|---|---|---|
| **T1** | Daily hormone regression | `log1p(PdG)`, `log1p(E3G)`, per-participant calibration-cycle normalized | within-participant Spearman ρ; MAE |
| **T1-NC** | Dilution negative control | PdG on **follicular days only** | ρ (should be ≈ 0 — bounds the urine-dilution artifact) |
| **T2-R** | Retrospective ovulation localization | ovulation day | MAE (days); **Skill over Calendar** |
| **T2-P** | Prospective ovulation nowcast | days-until-ovulation by lead time L | Skill over Calendar @ L |
| **T2-A** | Anovulation detection | binary, cycle-level | PR-AUC (calendar = prevalence by construction) |
| **T3** | 4-phase classification | {menstruation, late_follicular, ovulation, luteal} | macro-F1; per-class F1; ECE (10 equal-mass bins) |

**Ground-truth ovulation.** Synthetic: generator metadata. Real: LH-surge day (peak above an absolute
floor and a multiple of the cycle median), with a sustained-PdG-rise corroboration when PdG is sampled
around the candidate. PdG is measured on ~35% of mcPHASES days, so a PdG-mandatory rule is not used.

## Baselines (the calendar must be strong)
- `b0` marginal · `b1a` fixed day-14 (strawman) · `b1b` population-template · **`b1c` empirical-Bayes
  calendar-hazard** (participant random effect + shrinkage + conditional hazard update) — the **skill
  denominator** · `b2.5` diary-only · `b3` LightGBM · `b4` reference model.

`b1_retrospective` (next-onset − 14) is reported but is **not** a real-world baseline: it needs the
observed next onset, which a user does not have when she wants to know. The model's retrospective
confirmation is therefore scored against the *prospective* `b1c` — the guess a period app actually shows.

## Metric — Skill over Calendar
```
SoC = 1 − (Σ_i w_i Σ_t e^model_{i,t}) / (Σ_i w_i Σ_t e^b1c_{i,t}),   w_i = 1/n_days_i
D_i = MAE_b1c(i) − MAE_model(i)   (days; the primary endpoint statistic)
```
Report `median(D_i)`, a cluster-bootstrap 95% CI over participants (B=2000), and a Wilcoxon signed-rank
p. `mean(per-participant skill ratio)` is never reported.

## Splits
`RepeatedStratifiedGroupKFold`, k=6, seeds `[0,1]`, grouped on `participant_id`, stratified on
regularity (`range(cycle_len) ≥ 9` OR any cycle `<24` or `>38` days). OOF predictions averaged across
repeats before scoring. No participant spans a fold.

## Pre-registered primary endpoint
T2-R, irregular stratum, `median(D_i)` vs `b1c`, Wilcoxon, α=0.05 — committed in
`docs/preregistration.md` before any real-data run. All ~100 other cells are `exploratory`
(`is_primary=false` in the results JSON).

## Mandatory subgroup reporting
The harness reports every task stratified by regularity (primary), and additionally by BMI / age band
where available. Making stratified reporting mandatory is itself part of the specification.

## Results artifact
One JSON per run (`results/*.json`): schema version, tier, model, git commit, seeds, per-task and
per-stratum `MetricBlock`s with `is_primary`, CI bounds, `integrity_status`, wall-clock. The leaderboard
is built by reading these.

## Reproducing Tier B
Obtain mcPHASES v1.0.0 under your own PhysioNet DUA. The loader verifies file SHA-256 against
`configs/splits/mcphases_v1.json`; reproduce the exact splits by hashing your participant IDs with salt
`infradian-v1`. We publish no raw data.
