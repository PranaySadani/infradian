# Model Card: infradian-ref-s

Following Mitchell et al., "Model Cards for Model Reporting."

## Model details
- **Name:** infradian-ref-s (INFRADIAN reference, synthetic-trained).
- **Architecture:** three LightGBM models, a 4-class phase classifier and two hormone regressors
  (PdG, E3G), plus a median-filter + one-argmax ovulation decoder. Shallow, strongly regularized
  (num_leaves 15, min_child_samples 30, reg_lambda 1.0); hyperparameters fixed and tuned on synthetic
  data only, so there is zero selection leakage on real data.
- **Inputs:** causal windowed features from wearable channels (relative skin temperature, resting HR,
  RMSSD, respiratory rate, sleep efficiency) plus observed-menses-onset calendar features. All features
  are backward-looking (future-perturbation tested).
- **Version / license:** 0.1.0 / **Apache-2.0**.

## Training data
- **Synthetic only (INFRADIAN-SYNTH-1K, CC-BY-4.0).** This is deliberate: mcPHASES is DUA-restricted, and
  a checkpoint trained on restricted health data is a legal grey zone, so we do not distribute one. The
  synthetic-trained checkpoint is the only one we publish, which is why sim-to-real transfer is a
  structural result rather than a side experiment.

## Evaluation
See `results/*.json` and the README table. Headline: on the synthetic tier the model beats a strong
calendar baseline on irregular-cycle ovulation timing (SoC +0.40); on real mcPHASES the primary endpoint
is null (SoC −0.07, p=0.48). Reported with cluster-bootstrap CIs and a pre-registered primary endpoint.

## Intended use
Research infrastructure for benchmarking wearable-derived cycle inference. A reference point to beat.

## Out-of-scope use (important)
- **Not** a medical device; **not** diagnostic; **not** contraceptive or fertility guidance. Wearable-
  derived cycle estimates are not reliable enough to prevent or plan pregnancy.
- Do not apply to perimenopausal, PCOS-typical, or populations outside the training distribution (ages
  18–29, Canadian) and expect calibrated behavior.

We considered a behavioral-use license (OpenRAIL) and rejected it: it is non-OSI and unenforceable in
practice, and it blocks legitimate reuse. Misuse is addressed by this card's out-of-scope section, the
app's non-diagnostic banner, and the LLM refusal layer, not by license text.

## Ethical considerations
The genuine harm vector is misreading cycle estimates as contraception. The explanation layer hard-
refuses contraception and diagnosis questions; the app surfaces "not a diagnostic device" structurally.
See `ETHICS`-relevant notes in `LIMITATIONS.md`.
