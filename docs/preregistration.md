# INFRADIAN-BENCH v1.0 — Pre-registration

**Written and committed before any model was fit to real (Tier B) data.** The git history is
the timestamp: `git log --follow docs/preregistration.md` predates the first `results/*mcphases*`
artifact. This exists to prevent the multiplicity failure (plan §6.2): the benchmark reports
~80–120 metric cells (tasks × lead times × strata × modality × diary mode), and without a fixed
primary endpoint one could narrate whichever cell came out favourable. It commits to **two**
primary endpoints and nothing else. Every other cell is `exploratory`.

## Primary endpoints (exactly two)

### PRIMARY-C — always runs (Tier C synthetic)
- **Task:** T2-R (retrospective ovulation-day localization).
- **Stratum:** irregular cycles (`range(cycle_len) ≥ 9 d` OR any cycle `<24` or `>38 d`).
- **Statistic:** median per-participant paired difference `D_i = MAE_b1c(i) − MAE_model(i)`, in days,
  where `b1c` is the `calendar_hazard` baseline.
- **Test:** Wilcoxon signed-rank, two-sided, α = 0.05.
- **Uncertainty:** cluster bootstrap 95% CI over participants (B=2000).
- **Direction of interest:** `D_i > 0` means the model beats the strong calendar baseline.

### PRIMARY-B — conditional (Tier B, real mcPHASES)
- **Identical statistic** to PRIMARY-C, computed on mcPHASES.
- **Runs only if** the mcPHASES go/no-go criterion passes (plan §7.4): ≥1,500 participant-days
  with a hormone reading and ≥3 wearable channels, visible periodic LH spikes in ≥2/3 sanity
  participants, and confirmed units. If the go/no-go fails, PRIMARY-B is reported as
  **"not evaluated — data access/quality gate not met"** and is not silently dropped.

## Secondary (pre-specified, not primary)
- T2-A anovulation detection (PR-AUC vs prevalence baseline). *This is the fallback headline
  if the primary endpoints are null — the calendar cannot detect anovulation by construction.*
- T1 PdG regression: within-participant Spearman ρ, and SoC vs `b1c` template.
- T1-NC dilution negative control: within-participant ρ of PdG on follicular days only
  (expected ≈ 0; a large value bounds the urine-dilution artifact).
- T3 four-phase classification: macro-F1.
- Sim-to-real transfer: SoC of the synthetic-trained model on Tier B minus on Tier C.

## Everything else is exploratory
All per-lead-time cells (`SoC@L` for L ∈ {−5…+5}), all BMI/age-band strata, all modality
(`fitbit` vs `fitbit_cgm`) and diary-mode arms are labelled `"is_primary": false` in the
results JSON and described as exploratory in the README. They generate hypotheses; they do not
support confirmatory claims.

## Analysis decisions fixed in advance
- Splits: `RepeatedStratifiedGroupKFold`, k=6, seeds `[0,1]`, grouped on `participant_id`,
  stratified on regularity. OOF predictions averaged across repeats before scoring.
- No participant appears in more than one fold. The 20 dual-round mcPHASES participants are
  grouped by `participant_id` (not `segment_id`), so they never straddle a fold.
- `mean(S_i)` (mean of per-participant skill ratios) is never reported — only `median(D_i)` and
  the pooled participant-macro `SoC`.
