# INFRADIAN: submission pack

Everything below is ready to paste into the Hack-Nation submission form.

---

## 1. Project Summary (287 words)

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

I ran it on real clinical data (mcPHASES, 42 participants, 192 cycles) and on a CC-BY synthetic
cohort I generated so anyone can reproduce every number with no data-use agreement. On synthetic
data the model beats the calendar exactly where the calendar fails, irregular cycles, by 0.40 skill.
On real consumer wearables the pre-registered primary endpoint is a null, p equals 0.48. That gap is
the finding, and I lead with it.

A multimodal layer closes the loop from a person to the benchmark. Free-text or spoken journal
entries are mapped to INFRADIAN-SYM, a 17-code open symptom vocabulary, and a photo of an at-home
hormone test is transcribed into a numeric anchor for the inferred curve. Every model surface is
guarded: refusals for diagnosis, treatment and contraception run before any model call, the vision
reader returns a closed enum rather than prose, and explanations are rendered from typed slots so a
hallucinated number is structurally impossible. Forty-one red-team regression tests hold that line.

Shipped: benchmark spec, harness, synthetic dataset, reference checkpoint, split manifest for gated
data, the multimodal intake layer, and a live app.

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

Open six tabs first so you never click through navigation on camera:
`/`, `/explorer` (select S003, all three layers on), `/skill`, `/journal`, `/impact`, `/methods`,
and the GitHub repo.

**[0:00 to 0:20] Landing page**

> Everyone knows the circadian rhythm. Almost nobody can name the infradian one, the 28-day cycle
> that half the planet runs on. Every period app on earth predicts it with calendar arithmetic. I
> built the benchmark that measures whether a consumer wearable can actually do better, and exactly
> where it cannot.

**[0:20 to 0:50] Explorer, model layer off**

> This is a participant with irregular cycles. Before I show you any prediction, look at the cycle
> rail along the bottom: 40 days, then 20, then 32, then 28. The calendar assumes 28 every time. On
> the ovulation row, the hollow calendar markers miss the true ovulation markers by days, every
> single cycle. That error is 7.5 days for this person.

**[0:50 to 1:20] Toggle the model layer on**

> Same participant, same days, one consumer smartwatch.

*(Stay silent for four or five seconds and let the curves and uncertainty bands draw in.)*

> The model reconstructs the hormone curves from wearable signals with calibrated uncertainty bands,
> and its ovulation markers land within a couple of days of truth. Error drops from 7.5 days to 5.0.

**[1:20 to 1:50] Skill page**

> Here is the honest version. I built the strongest calendar baseline I could, an empirical-Bayes
> hazard model, not a fixed day-14 strawman, and I made it the denominator on purpose. On the
> synthetic cohort the gain is concentrated exactly where the calendar fails: plus 0.40 skill on
> irregular cycles, near zero on regular ones. And on real consumer wearables, the pre-registered
> primary endpoint is a null. P equals 0.48. I am showing you my own negative result, on the
> results page, as the headline.

**[1:50 to 2:20] Journal page, the multimodal layer**

> A benchmark is only useful if a real person can feed it. So this page takes whatever someone
> actually has. Type or speak an entry in plain language.

*(Type or dictate: "Really bad cramps today and I barely slept.")*

> That maps to INFRADIAN-SYM, an open seventeen-code symptom vocabulary, so free text becomes
> benchmark-shaped columns. You can also photograph an at-home hormone test and it transcribes the
> printed number into an anchor for the inferred curve. It refuses to read a two-line strip, because
> reading one would mean inventing a number from how dark a line looks.

**[2:20 to 2:40] The explanation panel, then the unsafe question**

> Every explanation is grounded by construction. The language model is never allowed to emit a digit.
> It writes prose with typed slots, my code fills every number from the model output, so a
> hallucinated figure is structurally impossible, not just unlikely. And when you ask it to diagnose
> you, or to use this as contraception, watch what happens.

*(Click "Do I have PCOS?", then type a contraception question.)*

> It refuses both, and points you to a clinician. I want to be honest about that: my first version of
> this classifier had a regex bug that made it almost entirely inert. An adversarial review of my own
> code caught it. It is now forty-one regression tests.

**[2:40 to 3:00] Methods page, then GitHub**

> Forty-two Canadian participants, ages 18 to 29. This says nothing about menopause or PCOS
> populations, and I do not pretend otherwise. The clinical data is DUA-restricted so I redistribute
> none of it, only checksums and hashed split IDs. The only checkpoint I can legally publish is
> trained on my synthetic cohort, so I made that cohort good enough to run the whole benchmark on,
> and then measured exactly how much skill it keeps on real data. Everything is open. Take it, and
> beat it.

---

## 4. Tech Video script (about 3 minutes)

Have the repo open, plus a terminal, plus `/skill`.

**[0:00 to 0:25] The architecture in one breath**

> Three tiers, driven by licensing. Tier A is NHANES, US public domain. Tier B is mcPHASES, clinical
> wearable and hormone data under a restricted PhysioNet license. Tier C is a synthetic cohort I
> generate myself. That licensing constraint drove the entire architecture, and I will show you
> why that turned out to be the most interesting part.

**[0:25 to 1:00] Synthetic-first, and why**

> I could not redistribute mcPHASES, and a model trained on it is a legal grey zone. So I built
> everything against a synthetic cohort first: literature-shaped hormone curves coupled to wearable
> channels through effect-size ranges I verified against primary papers. That single decision turned
> the restricted dataset from a critical-path dependency into a leaf node. It also gave me a
> leakage-free hyperparameter tuning set, because I tune on synthetic and evaluate on real.
>
> One detail worth flagging: an early draft used a resting-heart-rate shift of 8 beats per minute and
> an SDNN heart-rate-variability figure. Both were wrong. The real luteal shift is 2 to 4 bpm, and
> wearables report RMSSD, not SDNN. I caught it, documented the rejected numbers in
> `docs/effect_sizes.md`, and every magnitude in the generator now traces to a primary table.

**[1:00 to 1:45] The rigor that is actually enforced**

> Leakage control is mechanical, not aspirational. Every feature passes a future-perturbation test:
> I corrupt all data after day t, recompute, and assert the feature at day t is unchanged. Splits are
> participant-disjoint, and the 20 participants who appear in two study rounds are grouped by identity
> so they can never straddle a fold.
>
> I pre-registered one primary endpoint and committed it to git before running anything on real
> data. You can verify the ordering in the commit history. Everything else out of roughly a hundred
> metric cells is labelled exploratory, so I cannot cherry-pick a lucky one.
>
> And there is a negative control. The urinary hormone assay is not creatinine-normalized, so
> hydration contaminates it, and hydration is predictable from the same wearable features. I predict
> progesterone using only follicular days, where the true value is flat. On real data that comes out
> at 0.02, so the signal I report is not a hydration artifact.

**[1:45 to 2:20] The model, the multimodal layer, and what guards it**

> The model is deliberately boring: LightGBM with shallow trees and strong regularization, because
> with 42 participants overfitting is the dominant risk. A median filter plus one-argmax-per-cycle
> decoder turns per-day phase probabilities into a single ovulation day.
>
> On top of that sits a multimodal intake layer. You can type or speak a journal entry and it maps to
> INFRADIAN-SYM, a seventeen-code open symptom vocabulary, so free text becomes benchmark-shaped
> columns. You can photograph an at-home hormone test and it is transcribed into a numeric anchor for
> the inferred curve.
>
> That layer is the part I would defend hardest, because every one of those surfaces is a place a
> model could say something unsafe. Explanations never contain a model-written digit: the model emits
> typed slots, my renderer substitutes values, and a verifier rejects any stray numeral. The vision
> reader returns a closed enum instead of prose, so it has nowhere to put a claim. A refusal
> classifier for diagnosis, treatment and contraception runs before any model call.
>
> I want to be specific about how I know that holds. An adversarial review of my own code found that
> my first refusal classifier was almost entirely inert: I had wrapped truncated word stems in a
> trailing word boundary, which can never match, and nine of ten unsafe probes walked straight
> through, including the exact contraception question this demo shows being blocked. My tests missed
> it because every case I had written happened to use a convenient whole word. That bug is now
> forty-one red-team regression tests, and they assert on what must be allowed as well as what must be
> refused, because a guard that refuses everything is broken in the other direction.

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

- Record straight against https://infradian.vercel.app. The header reads "API live": the inference
  API runs as a Vercel Python function on the same origin, so the live inference panel really does
  call the model. No local server needed.
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
