"""Multimodal AI routes, shared by the Vercel function and the Docker backend.

Kept as plain functions taking and returning dicts so both entry points can mount them without
either owning a web framework. Every route degrades: with no OpenAI key configured the app stays
fully usable through the deterministic extractor, the template explainer and manual value entry.

Safety posture, applied uniformly:
  - every free-text surface passes through the refusal classifier before any model call, so
    diagnosis, treatment and contraception questions never reach a model;
  - user text is labelled as data, never followed as instruction;
  - the vision reader transcribes and is stripped of interpretation;
  - numbers in explanations are rendered from the payload, never emitted by the model.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Restated on every journal response. Structured severities and schema fields look clinical, and
# a record that looks clinical should say what it is not.
_DISCLAIMER = (
    "Recorded as self-reported symptoms for cycle research. Not a diagnosis and not medical advice."
)

MAX_TEXT = 2000
MAX_IMAGE_BYTES = 3_500_000  # Vercel caps the request body around 4.5MB
MAX_AUDIO_BYTES = 3_500_000


def ai_status() -> tuple[int, dict]:
    from infradian.aio import openai_client as oai

    st = oai.status()
    st["features"] = {
        "grounded_explanation": True,  # template fallback always available
        "symptom_extraction": True,  # deterministic fallback always available
        "voice_transcription": st["enabled"],  # browser speech recognition is the fallback
        "test_strip_reading": st["enabled"],  # manual entry is the fallback
    }
    return 200, st


def symptom_vocabulary() -> tuple[int, dict]:
    from infradian.aio.symptoms import vocabulary_json

    return 200, vocabulary_json()


def journal_extract(body: dict) -> tuple[int, dict]:
    """Free text or transcribed speech into INFRADIAN-SYM codes."""
    from infradian.aio.symptoms import extract
    from infradian.llm import guard

    raw = body.get("text", "")
    if not isinstance(raw, str):
        return 400, {"error": "text must be a string"}
    full = raw.strip()
    if not full:
        return 400, {"error": "no text provided"}

    # The gate reads the WHOLE entry, then the entry is truncated for extraction. Doing it the other
    # way round meant an unsafe request placed after 2000 characters of filler was never seen by the
    # classifier at all, which turned the length limit itself into the bypass.
    category = guard.classify_refusal_deep(full)
    text = full[:MAX_TEXT]
    # Truncation used to be silent, so symptoms written past the limit were dropped with no
    # indication. In a health log the person has to be told that part of what they wrote was not read.
    truncated = len(full) > MAX_TEXT

    # Extract regardless, so a refusal never silently destroys what the person actually logged.
    ex = extract(text, prefer_model=bool(body.get("use_model", True)) and not category)
    if truncated:
        ex.notes.append(
            f"Entry was longer than {MAX_TEXT} characters. Only the first {MAX_TEXT} were read for "
            "symptoms, so anything after that was not recorded."
        )

    if category:
        # The symptoms still come back: someone writing "terrible cramps, should I take ibuprofen?"
        # has logged a real symptom, and answering "I can't advise on medication" while binning the
        # cramps is its own kind of failure. The question is declined; the observation is kept.
        return 200, {
            "refused": True,
            "category": category,
            "message": guard.REFUSAL_COPY[category],
            "text": ex.text,
            "vocab_version": ex.vocab_version,
            "notes": ex.notes,
            "note": "The question was not answered. Symptoms you described were still recorded.",
            "disclaimer": _DISCLAIMER,
            "symptoms": [
                {
                    "code": s.code,
                    "label": s.label,
                    "category": s.category,
                    "severity": s.severity,
                    "schema_field": s.schema_field,
                    "evidence": s.evidence,
                }
                for s in ex.symptoms
            ],
            "schema_fields": ex.to_schema_fields(),
        }
    return 200, {
        "refused": False,
        "text": ex.text,
        "source": ex.source,
        "vocab_version": ex.vocab_version,
        "notes": ex.notes,
        "symptoms": [
            {
                "code": s.code,
                "label": s.label,
                "category": s.category,
                "severity": s.severity,
                "schema_field": s.schema_field,
                "evidence": s.evidence,
            }
            for s in ex.symptoms
        ],
        "schema_fields": ex.to_schema_fields(),
        "disclaimer": _DISCLAIMER,
    }


def transcribe_audio(body: dict) -> tuple[int, dict]:
    """Base64 audio into text via Whisper. The browser falls back to on-device speech recognition."""
    from infradian.aio import openai_client as oai

    b64 = str(body.get("audio_base64", ""))
    if "," in b64[:80]:  # strip a data: URI prefix if the client sent one
        b64 = b64.split(",", 1)[1]
    if not b64:
        return 400, {"error": "no audio provided"}

    # Check the encoded length before decoding, so an oversized payload is rejected without
    # allocating it.
    if len(b64) > MAX_AUDIO_BYTES:
        return 413, {"error": "audio too large, keep clips under about 30 seconds"}

    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception:  # noqa: BLE001
        return 400, {"error": "audio is not valid base64"}

    if len(raw) > MAX_AUDIO_BYTES:
        return 413, {"error": "audio too large, keep clips under about 30 seconds"}
    if not oai.available():
        return 200, {
            "available": False,
            "text": "",
            "message": "No OpenAI key configured. Use the browser microphone fallback or type the entry.",
        }

    try:
        text = oai.transcribe(raw, filename=str(body.get("filename", "clip.webm")))
    except oai.OpenAIUnavailable as e:
        return 200, {"available": False, "text": "", "message": f"Transcription unavailable ({e})."}

    return 200, {"available": True, "text": text, "model": oai.TRANSCRIBE_MODEL}


def read_test_strip(body: dict) -> tuple[int, dict]:
    """Photo of an at-home hormone test into a transcribed value. Never interprets the result."""
    from infradian.aio.vision import read_strip

    uri = str(body.get("image_base64", "")).strip()
    if not uri:
        return 400, {"error": "no image provided"}
    if not uri.startswith("data:image/"):
        uri = "data:image/jpeg;base64," + uri
    if len(uri) > MAX_IMAGE_BYTES:
        return 413, {"error": "image too large, please downscale before sending"}

    return 200, read_strip(uri).as_dict()


def explain(body: dict) -> tuple[int, dict]:
    """Grounded natural-language explanation of a prediction. Numbers are rendered, never generated."""
    from infradian.llm.explain import ExplainPayload
    from infradian.llm.explain import explain as do_explain

    # The measured quantities have no safe default. Silently substituting zeros produced a fluent,
    # confidently worded explanation asserting "ovulation at day -1", "rank correlation of 0.00" and
    # "off by 0.0 days versus 0.0 days for the calendar", stamped grounded:true with citations. Every
    # number was technically traceable to the payload, which is exactly why this slipped through: the
    # grounding contract guarantees a number came from the payload, not that the payload was real. An
    # explanation of nothing must fail, not read like a result.
    required = (
        "rhr_delta_bpm",
        "temp_delta_c",
        "pdg_spearman",
        "model_ovulation_day",
        "calendar_mae_days",
        "model_mae_days",
    )
    missing = [k for k in required if body.get(k) is None]
    if missing:
        return 400, {
            "error": "missing measured values; this endpoint explains a real prediction and has no "
            "defaults for them",
            "missing": missing,
            "hint": "obtain these from POST /predict/trajectory, or see the example in the README",
        }

    try:
        payload = ExplainPayload(
            participant_id=str(body.get("participant_id", "user")),
            cycle_regularity=str(body.get("cycle_regularity", "regular")),
            rhr_delta_bpm=float(body["rhr_delta_bpm"]),
            temp_delta_c=float(body["temp_delta_c"]),
            pdg_spearman=float(body["pdg_spearman"]),
            model_ovulation_day=int(body["model_ovulation_day"]),
            calendar_mae_days=float(body["calendar_mae_days"]),
            model_mae_days=float(body["model_mae_days"]),
            top_feature=str(body.get("top_feature", "the temperature CUSUM"))[:80],
        )
    except (TypeError, ValueError):
        # The exception text is Python internals. The caller already knows which fields exist.
        return 400, {"error": "one or more fields could not be read as a number"}

    # Physiological bounds. Without these, 1e308 rendered as a 309-digit number inside a calm
    # sentence, and -500 °C was asserted as this participant's measured temperature shift. A value
    # being faithfully carried from the payload does not make it a possible measurement.
    bounds = {
        "rhr_delta_bpm": (-30.0, 30.0),
        "temp_delta_c": (-3.0, 3.0),
        "pdg_spearman": (-1.0, 1.0),
        "model_ovulation_day": (0, 200),
        "calendar_mae_days": (0.0, 200.0),
        "model_mae_days": (0.0, 200.0),
    }
    for field, (lo, hi) in bounds.items():
        v = getattr(payload, field if field != "model_ovulation_day" else "model_ovulation_day")
        if not (lo <= v <= hi) or v != v:  # v != v catches NaN
            return 400, {
                "error": f"{field} is outside the plausible range [{lo}, {hi}]",
                "value": v if v == v else "NaN",
            }

    question = str(body.get("question", "")).strip()[:500] or None
    exp = do_explain(payload, question=question)

    from infradian.llm.evidence import EVIDENCE_BY_ID

    return 200, {
        "text": exp.text,
        "source": exp.source,
        "grounded": exp.grounded,
        "slots_used": exp.slots_used,
        "warnings": exp.warnings,
        "citations": [
            {"id": c, "claim": EVIDENCE_BY_ID[c].claim, "source": EVIDENCE_BY_ID[c].source}
            for c in exp.citations
            if c in EVIDENCE_BY_ID
        ],
    }
