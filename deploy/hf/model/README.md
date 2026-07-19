---
license: apache-2.0
tags:
  - womens-health
  - menstrual-cycle
  - wearables
  - lightgbm
  - tabular
library_name: lightgbm
---

# infradian-ref-s

Reference model for the [INFRADIAN benchmark](https://github.com/infradian/infradian). Three LightGBM
models (4-phase classifier + PdG/E3G regressors) with a median-filter ovulation decoder.

**Trained on synthetic data only** (INFRADIAN-SYNTH-1K). This is deliberate: the real clinical data
(mcPHASES) is DUA-restricted, so the only checkpoint we can legally distribute is synthetic-trained.
Sim-to-real transfer is therefore a structural result, on real mcPHASES the transfer is weak
(macro-F1 0.28), which is exactly the gap the benchmark measures.

## Use
```python
import joblib
b = joblib.load("infradian_ref.joblib")
# b["phase_clf"], b["pdg_reg"], b["e3g_reg"], b["feature_columns"], b["arm"]
```
Serve through the SAME feature transform used in training (`infradian.features.build.build_features`)
to guarantee parity. See `src/infradian/api/main.py`.

## Out-of-scope
Not a medical device. Not diagnostic. Not contraceptive or fertility guidance. Do not apply outside the
training distribution (synthetic; calibrated to ages 18–29). See `MODEL_CARD.md`.
