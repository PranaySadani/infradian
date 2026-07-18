# Tier B — the gated clinical data (mcPHASES)

**Go/no-go decision: GO.** The loader and benchmark run end-to-end on real mcPHASES data. This
document records what a DUA-holding lab needs to reproduce our Tier B numbers, and exactly what we
publish about the data (nothing raw).

## What we do NOT publish
Per PhysioNet Restricted Health Data License 1.5.0: no raw rows, no per-participant features, no
per-participant predictions, no individual-trajectory figures, and no mcPHASES-trained checkpoint.
Every public figure and video frame uses a **synthetic** participant.

## What we DO publish (committed, leaks nothing)
- `src/infradian/data/mcphases.py` — the loader (code is ours).
- `configs/splits/mcphases_v1.json` — SHA-256 file checksums + **hashed** participant IDs per fold.
  A DUA holder hashes their own IDs with salt `infradian-v1` to reproduce our exact splits.
- Aggregate metrics only, minimum cell size ≥ 5.

## Reproducing Tier B (for DUA holders)
1. Obtain mcPHASES v1.0.0 from PhysioNet under your own signed DUA.
2. `uv run python -c "from infradian.data.mcphases import load_canonical; load_canonical('<path>')"`
3. The loader verifies file checksums against the manifest; a mismatch stamps `integrity_status`
   rather than hard-failing.

## Layout decisions (from the CP2 timebox)
- **Keys:** `id` + `day_in_study`. `study_interval` (2022 / 2024) is the round marker → segment_id
  is built directly from it (no need to detect the ~day-905 discontinuity).
- **Skin temperature:** `wrist_temperature.temperature_diff_from_baseline` — RELATIVE deviation,
  confirming the canonical schema's relative-channel choice. Aggregated to a daily mean.
- **Daily wearable tables used:** wrist_temperature, resting_heart_rate,
  heart_rate_variability_details (rmssd), respiratory_rate_summary, sleep_score. The multi-GB
  intraday tables (heart_rate, calories, distance, steps) are not read; `steps` is left NaN.
- **Hormones:** `lh`, `estrogen`→`e3g`, `pdg`. PdG is sampled on ~35% of days (luteal-concentrated),
  so ovulation is anchored on the **LH surge** with PdG as corroboration only when available.
- **Dilution:** no creatinine / specific-gravity column exists → the urine-dilution artifact is
  bounded by the **T1-NC follicular-only negative control**, not corrected.

## Coverage (aggregate, publishable)
42 participants, 62 segments, 5,659 participant-days. 5,339 days with a hormone reading; **4,988
days with a hormone reading and ≥3 wearable channels**. 90 cycles detected, 65 ovulatory,
27.8% anovulatory overall (0% in the regular stratum, 38.5% in the irregular stratum — consistent
with the biology the benchmark targets).
