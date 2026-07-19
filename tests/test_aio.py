"""Tests for the multimodal AI layer.

Two things need proving. First, that every surface behaves correctly with NO OpenAI key, because
that is the state the public demo runs in and the state a reviewer will reproduce. Second, that the
model-backed paths are safe, which is tested by mocking the HTTP layer rather than spending real
tokens, and specifically that a model returning something hostile or malformed cannot get that
through to the user.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

import _ai_routes as ai_routes  # noqa: E402

from infradian.aio import openai_client as oai  # noqa: E402
from infradian.aio import symptoms, vision  # noqa: E402


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    """Default every test to the no-key state; individual tests opt into a fake key."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("_INFRADIAN_DOTENV_LOADED", "1")  # do not read a real .env during tests


# --- deterministic extractor -----------------------------------------------------------

def test_extractor_handles_negation():
    ex = symptoms.extract_deterministic("No cramps at all today, and I am not stressed")
    codes = {s.code for s in ex.symptoms}
    assert "SYM.PAIN.CRAMP" not in codes
    assert "SYM.STRESS.HIGH" not in codes


def test_extractor_severity_does_not_bleed_across_clauses():
    ex = symptoms.extract_deterministic("mild cramps but severe fatigue")
    by = {s.code: s.severity for s in ex.symptoms}
    assert by["SYM.PAIN.CRAMP"] == 1
    assert by["SYM.ENERGY.FATIGUE"] == 4


def test_extraction_maps_onto_canonical_schema_columns():
    """The point of the vocabulary: a journal entry has to land in the columns the model consumes."""
    from infradian.data import canonical as C

    ex = symptoms.extract_deterministic("bad cramps, low mood, slept badly, on my period")
    fields = ex.to_schema_fields()
    assert set(fields).issubset(set(C.ALL_COLUMNS))
    assert fields["menses_reported"] == 1.0
    assert 0.0 <= fields["sleep_eff"] <= 1.0


def test_every_vocab_code_maps_to_a_real_schema_field_or_none():
    from infradian.data import canonical as C

    for c in symptoms.CODES:
        assert c.schema_field is None or c.schema_field in C.ALL_COLUMNS


# --- no-key degradation ----------------------------------------------------------------

def test_all_surfaces_work_without_a_key():
    assert oai.available() is False

    status = ai_routes.ai_status()[1]
    assert status["enabled"] is False
    assert status["features"]["symptom_extraction"] is True
    assert status["features"]["grounded_explanation"] is True

    _, journal = ai_routes.journal_extract({"text": "bad cramps and low mood"})
    assert journal["source"] == "deterministic"
    assert journal["symptoms"]

    _, strip = ai_routes.read_test_strip({"image_base64": "data:image/png;base64,iVBORw0KGgo="})
    assert strip["readable"] is False
    assert "manually" in strip["reason"].lower()

    _, audio = ai_routes.transcribe_audio({"audio_base64": "AAAA"})
    assert audio["available"] is False

    # A full payload: the measured values have no defaults, because zeros rendered as a confident
    # explanation of a prediction that was never made.
    _, exp = ai_routes.explain({
        "cycle_regularity": "irregular", "model_ovulation_day": 20,
        "rhr_delta_bpm": 2.5, "temp_delta_c": 0.3, "pdg_spearman": 0.8,
        "calendar_mae_days": 13.3, "model_mae_days": 4.4,
    })
    assert exp["source"] == "template"
    assert exp["grounded"] is True


def test_journal_refuses_diagnostic_questions_before_any_model_call():
    for q in ["Do I have PCOS?", "Is this endometriosis?", "Can I use this as birth control?"]:
        _, r = ai_routes.journal_extract({"text": q})
        assert r["refused"] is True, q
        assert r["symptoms"] == []


# --- model-backed paths, HTTP mocked ---------------------------------------------------

@pytest.fixture
def fake_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")


def test_model_extraction_drops_hallucinated_codes(monkeypatch, fake_key):
    """A model inventing a code must not be able to introduce it into the vocabulary."""
    monkeypatch.setattr(
        oai,
        "chat_json",
        lambda *a, **k: {
            "symptoms": [
                {"code": "SYM.PAIN.CRAMP", "severity": 3, "evidence": "bad cramps"},
                {"code": "SYM.TOTALLY.MADE.UP", "severity": 4, "evidence": "nope"},
            ]
        },
    )
    ex = symptoms.extract_with_openai("bad cramps")
    codes = {s.code for s in ex.symptoms}
    assert codes == {"SYM.PAIN.CRAMP"}


def test_model_extraction_clamps_out_of_range_severity(monkeypatch, fake_key):
    monkeypatch.setattr(
        oai, "chat_json",
        lambda *a, **k: {"symptoms": [{"code": "SYM.GI.BLOAT", "severity": 99}]},
    )
    ex = symptoms.extract_with_openai("bloated")
    assert ex.symptoms[0].severity == 4


def test_extraction_falls_back_when_model_unavailable(monkeypatch, fake_key):
    def boom(*a, **k):
        raise oai.OpenAIUnavailable("simulated outage")

    monkeypatch.setattr(oai, "chat_json", boom)
    ex = symptoms.extract_with_openai("bad cramps")
    assert ex.source == "deterministic"
    assert ex.symptoms
    assert any("unavailable" in n for n in ex.notes)


# --- vision safety ---------------------------------------------------------------------

def test_vision_strips_interpretive_language(monkeypatch, fake_key):
    """The reader transcribes. If the model editorialises, that text must not reach the user.

    Since the red-team pass, `legibility` is a closed enum rather than a free-text `reason`, so
    an interpretive sentence has nowhere to travel: it falls back to fixed copy.
    """
    monkeypatch.setattr(
        oai,
        "vision_json",
        lambda *a, **k: {
            "readable": True, "device_class": "quantitative_readout", "analyte": "LH",
            "analyte_text_verbatim": "LH", "value": 42.0, "unit": "mIU/mL", "confidence": 0.4,
            "legibility": "This is a positive LH surge, you are likely ovulating",
        },
    )
    r = vision.read_strip("data:image/jpeg;base64,AAAA")
    assert r.readable is True
    assert r.value == 42.0
    assert "ovulat" not in r.reason.lower()
    assert "surge" not in r.reason.lower()
    assert r.warnings


def test_vision_rejects_implausible_values(monkeypatch, fake_key):
    monkeypatch.setattr(
        oai,
        "vision_json",
        lambda *a, **k: {
            "readable": True, "device_class": "quantitative_readout", "analyte": "LH",
            "analyte_text_verbatim": "LH", "value": 99999, "confidence": 0.9,
            "legibility": "clear",
        },
    )
    r = vision.read_strip("data:image/jpeg;base64,AAAA")
    assert r.readable is False
    assert "misread" in r.reason


def test_vision_rejects_unsupported_analyte(monkeypatch, fake_key):
    monkeypatch.setattr(
        oai,
        "vision_json",
        lambda *a, **k: {
            "readable": True, "device_class": "quantitative_readout", "analyte": "GLUCOSE",
            "analyte_text_verbatim": "GLUCOSE", "value": 5.5, "confidence": 0.9,
            "legibility": "clear",
        },
    )
    assert vision.read_strip("data:image/jpeg;base64,AAAA").readable is False


def test_vision_response_always_carries_a_disclaimer():
    r = vision.StripReading(readable=True, analyte="LH", value=10.0, unit="mIU/mL", confidence=0.8)
    assert "not a diagnostic" in r.as_dict()["disclaimer"].lower()


def test_oversized_payloads_are_rejected():
    big = "A" * (ai_routes.MAX_IMAGE_BYTES + 10)
    status, _ = ai_routes.read_test_strip({"image_base64": "data:image/jpeg;base64," + big})
    assert status == 413

    status, _ = ai_routes.transcribe_audio({"audio_base64": "A" * (ai_routes.MAX_AUDIO_BYTES + 10)})
    assert status == 413
