# If the headline is null or negative

**Written at the start, not the end.** The submission must be strong even if wearable-derived
inference does not beat the calendar. A rigorous negative result plus a reusable benchmark is
genuinely good science, and the project is designed so that a null result is a finding, not a
failure.

## The framing that survives a null

> The benchmark's first result is a **frontier map**, not a win. Wearable-derived hormonal
> inference shows **no skill** over a properly specified calendar baseline at negative lead times,
> **positive skill** at lag 0 to +3 days, and **large skill on ovulation *occurrence*** — where the
> calendar cannot compete by construction. That is the physiologically expected pattern (the thermal
> shift is a *consequence* of ovulation, appearing 1–3 days after it), and it is the first time
> anyone has measured it against a strong baseline with participant-level confidence intervals.

## Why a null is expected in parts of the grid
The corpus luteum forms *after* ovulation and drives the progesterone-mediated temperature rise.
So a wearable literally cannot see ovulation coming more than ~1–2 days ahead. Prospective
forecasting at lead ≤ −2 d **should** show ~zero skill. Claiming otherwise would be the suspicious
result; showing the expected null is the credible one.

## Three fallback headlines, none of which need beating the calendar

1. **T2-A anovulation detection.** A calendar baseline assumes ovulation every cycle, so its
   recall on anovulatory cycles is zero by construction. The wearable model detects the *absence*
   of a sustained thermal/HR elevation. SoC here is large and physiologically sound. This is the
   strongest fallback and it targets exactly the PCOS/perimenopause populations the thesis is about.

2. **The decoder ablation.** A per-day classifier emits ~3.4 ovulation-labelled days per cycle; the
   median-filter + one-argmax decoder emits 1.0 and cuts retrospective MAE. A clean methods result,
   independent of any baseline.

3. **The b1b → oracle gap decomposition.** How much calendar failure is *irreducible* (you cannot
   know this cycle's length in advance) versus *addressable*. Nobody has published this decomposition;
   it requires no wearable result at all.

## What to do operationally
- Keep the hero figure (skill vs lead time by stratum) exactly as designed — it reads correctly
  whether the curve is above or below zero.
- Change the `/skill` headline copy from "we beat the calendar" to "we mapped where wearables do
  and do not beat the calendar."
- Rehearse the null version of the demo video. Delivering a clean null out-scores delivering a
  suspicious win.
