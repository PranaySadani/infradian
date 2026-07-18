# Demo video — shot-by-shot (target ~3:00)

Drive from **5 bookmarked URLs**, never live-click through nav. Screen capture + voiceover, no face.
Run the app locally with the DUA copy present so the real numbers are available, but every plotted
trajectory is synthetic — say so.

| Time | Screen (URL) | On screen | Narration |
|---|---|---|---|
| 0:00–0:15 | `/` | Full-bleed thesis; the +0.40 / −0.07 / 42 stat tiles fade in | "Everyone knows the circadian rhythm. Almost nobody can name the *infradian* one — the 28-day cycle half the planet runs on. Every period app predicts it with calendar arithmetic. We built the benchmark that measures whether a wearable can do better — and exactly where it can't." |
| 0:15–0:45 | `/explorer` → select **S003 (irregular)**, model **off** | The three hormone panels + the CYCLES rail reading **40d · 20d · 32d · 28d** | "This is a synthetic participant with irregular cycles. Look at the cycle rail — forty days, then twenty, then thirty-two. The calendar assumes twenty-eight. On the ovulation row, the hollow calendar marks miss the truth marks by days, every cycle." |
| 0:45–1:15 | toggle **model on** | Ribbons + model lines draw in; truth dots land inside the bands; the orange model ovulation marks snap near truth | *(pause 5s)* "Same participant, same days. The model reconstructs the hormone curves from wearable signals, with uncertainty bands — and its ovulation marks land close to the truth where the calendar could not." |
| 1:15–1:45 | `/skill` | The three regularity rows; green synthetic bars growing regular→irregular; muted real bars at ~0 | "We built the strongest calendar baseline we could — a hazard model, not a strawman — and made it the denominator on purpose. On the synthetic cohort the gain is concentrated exactly where the calendar fails: irregular cycles, plus-point-four skill. On **real** consumer wearables the pre-registered primary endpoint is a **null** — p equals point-four-eight — and we say so, right here." |
| 1:45–2:15 | `/explorer` scroll to the explanation panel; click the **"Do I have PCOS?"** chip | The claim-guard badge "N of N cited · 0 diagnostic · numbers never model-generated"; citation chips; then the refusal text | "The explanation is grounded by construction — the model literally cannot emit a number; it writes slots we fill from the model output, so every figure is traceable. And when you ask it to diagnose you — it refuses." |
| 2:15–2:45 | `/methods` + HuggingFace tab | The three-tier license table; the synth dataset card | "Anyone can run this today on the CC-BY synthetic cohort, with zero data access. The only checkpoint we can legally publish is trained on that synthetic data — so we measured exactly how much skill it keeps on real clinical data. That sim-to-real gap is the finding." |
| 2:45–3:00 | `/methods` limitations | The cohort card: n=42, ages 18–29 | "Forty-two Canadian adults — this says nothing about menopause or PCOS, and we don't pretend it does. Not a diagnostic device. Everything — benchmark, synthetic data, model, harness — is open. Take it. Beat it." |

**The three moves that win it:** (1) admitting the real-data primary is null, (2) having deliberately
strengthened our own baseline, (3) rendering the honesty — the claim-guard badge and the refusal — as
product, not fine print.
