"""INFRADIAN FastAPI backend. Real and wired, but deliberately OFF the demo critical path — the
frontend reads static JSON, so a backend outage cannot break the demo. The API powers the
interactive features: live prediction on an uploaded/synthetic series, grounded explanation, the
leaderboard, and synthetic-cohort sampling.

Loads the synthetic-trained reference bundle at startup (the only checkpoint we may distribute) and
serves predictions through the SAME feature transform used in training, guaranteeing parity.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from infradian.data import canonical as C
from infradian.features.build import build_features

BUNDLE_PATH = Path("results/models/infradian-ref-s/infradian_ref.joblib")
RESULTS = Path("results")

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-load the model bundle once (graceful if absent — the API still serves metadata/leaderboard).
    if BUNDLE_PATH.exists():
        from infradian.models.reference import load_bundle

        _state["bundle"] = load_bundle(BUNDLE_PATH)
    yield
    _state.clear()


app = FastAPI(title="INFRADIAN API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # static frontend on another origin
    allow_methods=["*"],
    allow_headers=["*"],
)


class DayRecord(BaseModel):
    day_in_study: int
    skin_temp_dev_c: float | None = None
    rhr_bpm: float | None = None
    hrv_rmssd_ms: float | None = None
    resp_rate: float | None = None
    sleep_eff: float | None = None
    steps: float | None = None
    menses_reported: int = 0
    cramps: float | None = None
    mood: float | None = None
    stress: float | None = None


class TrajectoryRequest(BaseModel):
    participant_id: str = "user"
    days: list[DayRecord] = Field(default_factory=list)


class TrajectoryResponse(BaseModel):
    participant_id: str
    days: list[int]
    pdg_pred: list[float]
    e3g_pred: list[float]
    ovulation_prob: list[float]
    model: str


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model_loaded": "bundle" in _state}


@app.get("/model")
def model_meta() -> dict:
    b = _state.get("bundle")
    return {
        "name": "infradian-ref-s",
        "trained_on": "synthetic (Tier C) only — the only legally distributable checkpoint",
        "license": "Apache-2.0",
        "arm": b["arm"] if b else None,
        "n_features": len(b["feature_columns"]) if b else 0,
    }


@app.post("/predict/trajectory", response_model=TrajectoryResponse)
def predict_trajectory(req: TrajectoryRequest) -> TrajectoryResponse:
    b = _state.get("bundle")
    if b is None:
        raise HTTPException(503, "reference bundle not loaded")
    if not req.days:
        raise HTTPException(400, "no days provided")

    rows = []
    for d in req.days:
        row = d.model_dump()
        row[C.KEY_PARTICIPANT] = req.participant_id
        row[C.KEY_SEGMENT] = C.make_segment_id(req.participant_id, 0)
        for col in C.ALL_COLUMNS:
            row.setdefault(col, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)[C.ALL_COLUMNS]

    fm = build_features(df, arm=b["arm"])
    # same columns as training => parity; coerce to float (all-None channels can be object dtype)
    X = fm[b["feature_columns"]].apply(pd.to_numeric, errors="coerce")
    pdg = np.expm1(b["pdg_reg"].predict(X))
    e3g = np.expm1(b["e3g_reg"].predict(X))
    prob = b["phase_clf"].predict_proba(X)
    ov_col = list(b["phase_clf"].classes_).index(C.PHASES.index("ovulation")) if C.PHASES.index("ovulation") in b["phase_clf"].classes_ else None
    ov = prob[:, ov_col] if ov_col is not None else np.zeros(len(X))

    return TrajectoryResponse(
        participant_id=req.participant_id,
        days=fm[C.KEY_DAY].astype(int).tolist(),
        pdg_pred=[round(float(v), 3) for v in pdg],
        e3g_pred=[round(float(v), 3) for v in e3g],
        ovulation_prob=[round(float(v), 4) for v in ov],
        model="infradian-ref-s",
    )


@app.get("/benchmark/leaderboard")
def leaderboard() -> list[dict]:
    p = RESULTS.parent / "web" / "public" / "data" / "leaderboard.json"
    if p.exists():
        return json.loads(p.read_text())
    return []


@app.get("/synthetic/sample")
def synthetic_sample(n: int = 1, days: int = 90, seed: int = 0) -> dict:
    """Sample a small synthetic cohort on demand (demonstrates the CC-BY generator via the API)."""
    from infradian.synth.generator import generate_cohort

    n = max(1, min(n, 20))
    df = generate_cohort(n=n, n_days=days, seed=seed)
    public = df[[c for c in df.columns if not c.startswith("_")]]
    return {"n": int(df[C.KEY_PARTICIPANT].nunique()), "rows": len(df), "sample": public.head(5).to_dict("records")}
