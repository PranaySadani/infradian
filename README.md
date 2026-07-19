# INFRADIAN

**An open benchmark for hormonal trajectory inference from consumer wearables.**
*The rhythm nobody benchmarks.*

**[Live app](https://infradian.vercel.app)** · **[Benchmark spec](BENCHMARK.md)** · **[Synthetic dataset (CC-BY)](datasets/infradian-synth-1k)** · **[Results](results/)** · **[Limitations](LIMITATIONS.md)**

Everyone knows the **circadian** rhythm. Far fewer people can name the **infradian** one — the ~28-day
endocrine cycle that shapes temperature, heart rate, sleep, mood, and metabolism for roughly half of
humanity. Period-tracking apps predict it with calendar arithmetic: count the days, guess the rest.
That works for textbook cycles and breaks precisely for the people it matters most for — irregular
cycles, PCOS, perimenopause, athletes, anyone under stress. Wearables have been streaming the
physiological signal all along. What has been missing is not data and not models — it is a **ruler**.

> The only checkpoint we can legally distribute is trained on **synthetic** data (the real clinical
> data is DUA-restricted). So we made the synthetic cohort good enough to run the whole benchmark on
> with zero access friction — and measured exactly how much skill it retains on real clinical wearable
> data. That **sim-to-real gap is the finding.**

---

## The headline, stated honestly

The benchmark's first result is a **frontier map**, not a victory lap. Every number is **skill over a
strong, adaptive calendar baseline** (an empirical-Bayes hazard model — not a day-14 strawman), and we
pre-registered a single primary endpoint before running anything on real data.

| Task | Metric | **Synthetic** (Tier C) | **Real mcPHASES** (Tier B) | Sim→Real transfer |
|---|---|---|---|---|
| Ovulation timing — **irregular cycles** *(pre-registered primary)* | Skill over Calendar | **+0.40** (95% CI 0.35–0.45, p<0.001) | **−0.07** (CI −0.22–0.05, p=0.48 — **null**) | — |
| Ovulation timing — regular cycles | Skill over Calendar | +0.05 | −0.05 | — |
| 4-phase classification | macro-F1 | 0.81 | **0.46** (chance = 0.25) | 0.28 |
| Daily PdG reconstruction | within-participant ρ | 0.88 | 0.16 | 0.08 |
| Dilution negative control | ρ (follicular only) | 0.30 | **0.02** | — |

**What this says.** On clean physiology the model beats the calendar exactly where the calendar fails —
irregular cycles — and barely elsewhere. On **real consumer wearables the primary endpoint is a null**:
the day-level signal-to-noise is too low to beat the calendar for ovulation timing, even though
cycle-phase information is clearly present (F1 0.46 ≫ 0.25 chance). The dilution negative control is
~0 on real data, so the little signal that exists is not a hydration artifact. **This is a rigorous,
pre-registered negative result plus a reusable instrument — which is good science, and more credible
than a suspicious win.**

---

## What you get (the reusable contribution)

- **`BENCHMARK.md`** — a frozen specification: tasks, the named **Skill over Calendar** metric,
  participant-disjoint splits, a calibration protocol, and mandatory subgroup reporting.
- **A runnable harness** (`infradian` Python package, Apache-2.0) with a mechanically-enforced causality
  contract (future-perturbation tests) and cluster-bootstrap confidence intervals.
- **[INFRADIAN-SYNTH-1K](datasets/infradian-synth-1k)** — a CC-BY synthetic longitudinal cohort so *anyone*
  can run the whole benchmark with no data-use agreement. 600 participants, 72,000 participant-days,
  participant-disjoint train/validation/test parquet splits, shipped in this repo. Wearable coupling is
  calibrated to *verified* published effect sizes (`docs/effect_sizes.md`), with a hydration nuisance term
  that makes the dilution control meaningful.
- **infradian-ref-s** — a reference checkpoint trained on synthetic data only (Apache-2.0).
- **A frozen split manifest + SHA-256 checksums** so any lab holding its own mcPHASES DUA reproduces our
  exact splits and produces directly comparable numbers — while we redistribute *none* of the raw data.
- **A grounded LLM explanation layer** whose numbers are impossible to hallucinate by construction, with
  a refusal layer and a shipped eval.
- **A live app** (`web/`) and a **FastAPI backend** wired to the reference model.

---

## Quickstart — reproduce every synthetic number with no data access

```bash
# needs: uv (https://docs.astral.sh/uv), Python 3.12 (uv fetches it)
make setup            # create the environment + install the git privacy hook
make synth            # generate INFRADIAN-SYNTH-1K (CC-BY)
make eval             # run the benchmark, write results/*.json
uv run python -m infradian.llm.eval          # the explanation-layer eval (12/12)
uv run python -m pytest                       # leakage, causality, parity, privacy, API tests
```

`make reproduce` runs the whole synthetic pipeline end to end. **None of it needs mcPHASES** — that is
what the synthetic tier buys.

### With a mcPHASES DUA (Tier B)

```bash
# after obtaining mcPHASES v1.0.0 from PhysioNet under your own signed DUA:
uv run python scripts/run_full_bench.py --mcphases /path/to/mcphases-1.0.0
```

The loader verifies file checksums against `configs/splits/mcphases_v1.json` and stamps
`integrity_status` on any mismatch (it does not silently pass or hard-fail).

### Run the app

```bash
cd web && npm install && npm run build   # static export -> web/out (deploy anywhere)
uv run make serve                         # optional FastAPI backend on :8000
```

---

## The three-tier, license-correct design

| Tier | Contents | License | Published? |
|---|---|---|---|
| **A — open** | NHANES-derived hormone tables | CC0 (US public domain) | data + code |
| **B — gated** | mcPHASES clinical wearable+hormone data | PhysioNet RHD 1.5.0 | **loader + checksums + hashed splits only** |
| **C — synthetic** | INFRADIAN-SYNTH-1K | CC-BY-4.0 | data + generator |

**mcPHASES handling:** no raw rows, no per-participant figures, no mcPHASES-trained checkpoint. Every
figure in the app and every video frame plots a *synthetic* participant. A pre-commit hook blocks any
data commit; the split manifest stores salted hashes of participant IDs, never the IDs.

---

## Rigor, in one place

- **Pre-registration** (`docs/preregistration.md`, committed before any real-data run): one primary
  endpoint, everything else labelled exploratory — no cherry-picking one lucky cell out of ~100.
- **Strong baseline on purpose:** the denominator is an empirical-Bayes calendar-hazard model, so a win
  means something. We report where we *don't* beat it.
- **Causality by construction:** every feature passes a future-perturbation + truncation-invariance test.
- **Participant-disjoint splits:** the 20 dual-round mcPHASES participants never straddle a fold.
- **Honest uncertainty:** cluster-bootstrap CIs over participants; a paired Wilcoxon on per-participant
  differences; `mean(skill ratio)` is never reported.
- **Verified effect sizes:** every generator magnitude is traced to a primary table, with rejected wrong
  numbers documented (`docs/effect_sizes.md`).

## Repository layout

```
src/infradian/      data (canonical schema, mcphases loader, manifest), features (causal),
                    synth (generator), bench (splits, baselines, metrics, runner), models (gbm,
                    reference), llm (evidence, guard, explain, eval), api (FastAPI)
scripts/            run_bench.py, run_full_bench.py, export_web_data.py
configs/splits/     mcphases_v1.json  (checksums + hashed splits — committed, leaks nothing)
docs/               effect_sizes.md, preregistration.md, if_null.md, tier_b.md
results/            *.json  (published metrics) + models/infradian-ref-s/feature_spec.json
web/                Next.js app (static export)
tests/              splits, causality, privacy, model, api, llm, synthetic-effect-sizes
```

## Not

Not a diagnostic device. Not contraceptive or fertility guidance. The clinical cohort is 42 Canadian
adults aged 18–29 — it says **nothing** about perimenopause, PCOS-typical populations, or older or more
diverse groups. See `LIMITATIONS.md`.

## Licenses

Code **Apache-2.0** · synthetic data **CC-BY-4.0** · NHANES-derived data **CC0** · reference weights
**Apache-2.0** · mcPHASES **not redistributed**.

## Data

- mcPHASES: Lin et al., *Scientific Data* 13:411 (2026), DOI [10.13026/zx6a-2c81](https://doi.org/10.13026/zx6a-2c81) (restricted access).
- NHANES: CDC/NCHS, US public domain.
