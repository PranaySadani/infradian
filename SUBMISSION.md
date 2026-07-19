# INFRADIAN: submission pack

Everything below is ready to paste into the Hack-Nation submission form.

---

## 1. Project Summary (238 words)

Everyone knows the circadian rhythm. Almost nobody can name the infradian one, the roughly 28-day
endocrine cycle that shapes temperature, heart rate, sleep and metabolism for half the planet. Every
period-tracking app predicts it with calendar arithmetic, which works for textbook cycles and breaks
precisely for the people it matters most for: irregular cycles, PCOS, perimenopause, athletes.
Wearables have been streaming the physiological signal all along. What has been missing is not data
and not models. It is a ruler.

INFRADIAN is that ruler: an open benchmark for inferring daily hormonal state from consumer
wearables, scored on one question, does it beat the calendar, and for whom. It defines six tasks, a
named metric (Skill over Calendar), participant-disjoint splits, and a causality contract enforced by
tests rather than asserted in prose. The calendar baseline is deliberately strong, an empirical-Bayes
hazard model, so a win means something.

We ran it on real clinical data (mcPHASES, 42 participants, 192 cycles) and on a CC-BY synthetic
cohort we generated so anyone can reproduce every number with no data-use agreement. On synthetic
data the model beats the calendar exactly where the calendar fails, irregular cycles, by 0.40 skill.
On real consumer wearables the pre-registered primary endpoint is a null, p equals 0.48. That gap is
the finding, and we lead with it.

Shipped: benchmark spec, harness, synthetic dataset, reference checkpoint, split manifest for gated
data, a grounded explanation layer that cannot hallucinate a number, and a live app.

---

## 2. Links

| Field | Value |
|---|---|
| GitHub Repository | https://github.com/PranaySadani/infradian |
| Live application | https://infradian.vercel.app |
| Dataset | https://github.com/PranaySadani/infradian/tree/main/datasets/infradian-synth-1k (CC-BY-4.0, generated). Source clinical data: mcPHASES, DOI 10.13026/zx6a-2c81, restricted access, not redistributed. |
| Zipped Code | `infradian-submission.zip` in your Downloads folder |

---

## 3. Demo Video script (about 3 minutes)

Open five tabs first so you never click through navigation on camera:
`/`, `/explorer` (select S003, all three layers on), `/skill`, `/methods`, and the GitHub repo.

**[0:00 to 0:20] Landing page**

> Everyone knows the circadian rhythm. Almost nobody can name the infradian one, the 28-day cycle
> that half the planet runs on. Every period app on earth predicts it with calendar arithmetic. We
> built the benchmark that measures whether a consumer wearable can actually do better, and exactly
> where it cannot.

**[0:20 to 0:50] Explorer, model layer off**

> This is a participant with irregular cycles. Before I show you any prediction, look at the cycle
> rail along the bottom: 40 days, then 20, then 32, then 28. The calendar assumes 28 every time. On
> the ovulation row, the hollow calendar markers miss the true ovulation markers by days, every
> single cycle. That error is 7.5 days for this person.

**[0:50 to 1:25] Toggle the model layer on**

> Same participant, same days, one consumer smartwatch.

*(Stay silent for four or five seconds and let the curves and uncertainty bands draw in.)*

> The model reconstructs the hormone curves from wearable signals with calibrated uncertainty bands,
> and its ovulation markers land within a couple of days of truth. Error drops from 7.5 days to 5.0.

**[1:25 to 2:00] Skill page**

> Here is the honest version. We built the strongest calendar baseline we could, an empirical-Bayes
> hazard model, not a fixed day-14 strawman, and we made it the denominator on purpose. On the
> synthetic cohort the gain is concentrated exactly where the calendar fails: plus 0.40 skill on
> irregular cycles, near zero on regular ones. And on real consumer wearables, the pre-registered
> primary endpoint is a null. P equals 0.48. We are showing you our own negative result, on the
> results page, as the headline.

**[2:00 to 2:30] Explanation panel, then click the unsafe question**

> The explanation is grounded by construction. The language model is never allowed to emit a digit.
> It writes prose with typed slots, and our code fills every number from the model output, so a
> hallucinated figure is structurally impossible, not just unlikely. Every claim carries a citation
> chip. And when you ask it to diagnose you, watch what happens.

*(Click "Do I have PCOS?")*

> It refuses, and points you to a clinician.

**[2:30 to 3:00] Methods page, then GitHub**

> Forty-two Canadian participants, ages 18 to 29. This says nothing about menopause or PCOS
> populations, and we do not pretend otherwise. The clinical data is DUA-restricted so we redistribute
> none of it, only checksums and hashed split IDs. The only checkpoint we can legally publish is
> trained on our synthetic cohort, so we made that cohort good enough to run the whole benchmark on,
> and then measured exactly how much skill it keeps on real data. Everything is open. Take it, and
> beat it.

---

## 4. Tech Video script (about 3 minutes)

Have the repo open, plus a terminal, plus `/skill`.

**[0:00 to 0:25] The architecture in one breath**

> Three tiers, driven by licensing. Tier A is NHANES, US public domain. Tier B is mcPHASES, clinical
> wearable and hormone data under a restricted PhysioNet license. Tier C is a synthetic cohort we
> generate ourselves. That licensing constraint drove the entire architecture, and I will show you
> why that turned out to be the most interesting part.

**[0:25 to 1:00] Synthetic-first, and why**

> We could not redistribute mcPHASES, and a model trained on it is a legal grey zone. So we built
> everything against a synthetic cohort first: literature-shaped hormone curves coupled to wearable
> channels through effect-size ranges we verified against primary papers. That single decision turned
> the restricted dataset from a critical-path dependency into a leaf node. It also gave us a
> leakage-free hyperparameter tuning set, because we tune on synthetic and evaluate on real.
>
> One detail worth flagging: an early draft used a resting-heart-rate shift of 8 beats per minute and
> an SDNN heart-rate-variability figure. Both were wrong. The real luteal shift is 2 to 4 bpm, and
> wearables report RMSSD, not SDNN. We caught it, documented the rejected numbers in
> `docs/effect_sizes.md`, and every magnitude in the generator now traces to a primary table.

**[1:00 to 1:45] The rigor that is actually enforced**

> Leakage control is mechanical, not aspirational. Every feature passes a future-perturbation test:
> we corrupt all data after day t, recompute, and assert the feature at day t is unchanged. Splits are
> participant-disjoint, and the 20 participants who appear in two study rounds are grouped by identity
> so they can never straddle a fold.
>
> We pre-registered one primary endpoint and committed it to git before running anything on real
> data. You can verify the ordering in the commit history. Everything else out of roughly a hundred
> metric cells is labelled exploratory, so we cannot cherry-pick a lucky one.
>
> And there is a negative control. The urinary hormone assay is not creatinine-normalized, so
> hydration contaminates it, and hydration is predictable from the same wearable features. We predict
> progesterone using only follicular days, where the true value is flat. On real data that comes out
> at 0.02, so the signal we report is not a hydration artifact.

**[1:45 to 2:20] The model and the LLM layer**

> The model is deliberately boring: LightGBM with shallow trees and strong regularization, because
> with 42 participants overfitting is the dominant risk. A median filter plus one-argmax-per-cycle
> decoder turns per-day phase probabilities into a single ovulation day.
>
> The explanation layer is the part I would defend hardest. The language model never emits a digit.
> It returns prose containing typed slots, our renderer substitutes values from the model output, and
> a verifier rejects any stray numeral. Hallucinated numbers are structurally impossible. There is a
> refusal classifier for diagnosis, treatment and contraception questions, and a red-team eval that
> ships in the repo at 12 out of 12.

**[2:20 to 3:00] Stack, and the result**

> Python with uv, LightGBM, FastAPI for the backend, Next.js static export on Vercel for the
> frontend, with a hand-rolled d3 SVG chart. The demo path is entirely static JSON, so a backend
> outage cannot break it, while the live inference panel proves the wiring end to end.
>
> The headline: on synthetic data, plus 0.40 skill on irregular cycles. On real consumer wearables,
> a null. Cycle-phase information is clearly present, macro-F1 0.46 against 0.25 chance, but the
> day-level signal-to-noise is not there yet for ovulation timing. That is a real result, it is
> pre-registered, and the benchmark is the instrument that made it measurable.

---

## 5. Team Video script (about 60 to 90 seconds)

Replace the bracketed parts with your own details.

> Hi, I'm [name], and I built INFRADIAN solo for Challenge 05.
>
> [One or two sentences on your background and what draws you to this: ML engineering, health data,
> open science, whatever is true.]
>
> I picked this challenge because the brief said something I agreed with: the next generation of AI
> for women's health will not come from one company, it will come from shared datasets and
> reproducible benchmarks. So I deliberately did not build another prediction app. I built the
> measurement layer underneath one.
>
> The hardest decision was what to do when the result came back null. On real wearable data, the
> model does not beat the calendar. I could have found a cell in the results that looked good and led
> with that. Instead I pre-registered a single endpoint before touching the real data, and put the
> null on the results page as the headline. A benchmark that only reports wins is not measuring
> anything.
>
> Everything is open: the spec, the harness, the synthetic dataset, the model, and the limitations.
> Thanks for watching.

---

## 6. Recording notes

- Run the app locally so the header reads "API live": `make serve` in one terminal, then
  `cd web && npm run build && npx serve out -l 3210` in another. Or record against
  https://infradian.vercel.app, where the header will read "API offline", which is a normal state
  because every view reads static JSON. Mention that if it is on screen.
- Every trajectory shown anywhere is a synthetic participant. Real clinical data appears only as
  aggregate metrics. Say this once, on camera, during the demo.
- Screen capture at 1440 by 900 or larger. The layout is designed for a laptop-width viewport.
- Two takes maximum. A clean 80 percent take beats a stitched perfect one.

---

## 7. Optional: mirror the artifacts to HuggingFace

Not required. The dataset and model already ship in the repo, so the submission is complete without
this. Mirroring just adds two more citable links.

Get a **write** token from https://huggingface.co/settings/tokens, then pick one of three ways to
supply it. The token is never passed as a command-line argument, so it stays out of your shell
history and out of `ps`.

```bash
uv pip install huggingface_hub

# option 1: this shell only
export HF_TOKEN=hf_xxx

# option 2: persists, .env is gitignored and the pre-commit hook blocks it
cp .env.example .env && $EDITOR .env        # set HF_TOKEN=hf_xxx

# option 3: interactive login
huggingface-cli login
```

Then:

```bash
make publish-hf-check HF_ORG=<your-hf-username>   # verify artifacts + token, uploads nothing
make publish-hf       HF_ORG=<your-hf-username>   # publish
```

This creates `<you>/infradian-synth-1k` (CC-BY-4.0 parquet, the dataset viewer works automatically)
and `<you>/infradian-ref-s` (Apache-2.0 weights). Add `--private` to the script call if you want to
inspect them before making them public. Add both links to the submission form if you do this.
