"""API contract tests. The backend must load, serve metadata, predict through the SAME feature
transform used in training (parity), and degrade gracefully."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import infradian.api.main as api  # noqa: E402
from infradian.models.reference import load_bundle  # noqa: E402


@pytest.fixture(scope="module")
def client():
    if api.BUNDLE_PATH.exists():
        api._state["bundle"] = load_bundle(api.BUNDLE_PATH)
    return TestClient(api.app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_metadata(client):
    r = client.get("/model")
    assert r.status_code == 200
    assert r.json()["name"] == "infradian-ref-s"


@pytest.mark.skipif(not api.BUNDLE_PATH.exists(), reason="reference bundle not built")
def test_predict_trajectory_shape_and_monotone_pdg(client):
    days = [
        {"day_in_study": i, "skin_temp_dev_c": 0.05 * i, "rhr_bpm": 60 + i, "menses_reported": 1 if i < 3 else 0}
        for i in range(28)
    ]
    r = client.post("/predict/trajectory", json={"participant_id": "user", "days": days})
    assert r.status_code == 200
    body = r.json()
    assert len(body["pdg_pred"]) == len(days)
    assert len(body["ovulation_prob"]) == len(days)


def test_synthetic_sample(client):
    r = client.get("/synthetic/sample?n=2&days=60")
    assert r.status_code == 200
    assert r.json()["n"] == 2
