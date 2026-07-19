"""Grounded explanation engine.

Given a structured payload (a participant's model output for a day/cycle), produce a plain-language
explanation whose every number is traceable by construction. Two paths:

  - LLM path (if OPENAI_API_KEY is set): the model returns a slot-only template; we verify no free
    numbers, verify all slots are known, then deterministically render values from the payload.
  - Template path (default, and the demo path): a deterministic templated explanation. Fully offline,
    cannot fail, and used to warm the demo cache. The app works with no API key.

Prompts are versioned files under prompts/ and hashed into the output for reproducibility.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from infradian.llm import evidence, guard

PROMPT_DIR = Path(__file__).parent / "prompts"
MODEL = "gpt-4o-mini"  # cheap, structured-output capable; overridable via INFRADIAN_LLM_MODEL


@dataclass
class ExplainPayload:
    participant_id: str
    cycle_regularity: str
    rhr_delta_bpm: float
    temp_delta_c: float
    pdg_spearman: float
    model_ovulation_day: int
    calendar_mae_days: float
    model_mae_days: float
    top_feature: str


@dataclass
class Explanation:
    text: str
    citations: list[str]
    slots_used: list[str]
    grounded: bool
    source: str  # "llm" | "template" | "refusal"
    prompt_hash: str = ""
    warnings: list[str] = field(default_factory=list)


# The two free-text slots are attacker-controlled if the endpoint is called directly, so they are
# constrained here rather than trusted. Without this, a POST could put arbitrary medical prose into
# `cycle_regularity` or `top_feature` and have it rendered back inside an explanation stamped
# "grounded", with citations, on the deterministic path that needs no API key at all. The slot and
# digit checks run on the model's TEMPLATE, never on substituted values, so they cannot catch it.
_ALLOWED_REGULARITY = {"regular", "irregular", "highly irregular", "unknown"}
_SAFE_FEATURE_RE = re.compile(r"^[a-zA-Z0-9 ,\-]{1,60}$")
_DEFAULT_FEATURE = "the temperature CUSUM"


# Each slot renders WITH its unit already attached ("+2.5 bpm", "day 14"). A model told only the slot
# names reasonably assumes the slot is a bare number and writes the unit itself, producing
# "+2.5 bpm beats per minute" or "on day day 14". The system prompt now says so explicitly, but a
# prompt instruction is a request, not a guarantee, so the renderer also repairs it. Discovered when a
# real API key was first configured in production: until then every explanation took the deterministic
# template path, where this could not occur.
_REDUNDANT_AFTER = {
    "rhr_delta": ("beats per minute", "beats/min", "bpm"),
    "temp_delta": ("degrees celsius", "degrees c", "degrees", "celsius", "°c"),
    "calendar_mae": ("days", "day"),
    "model_mae": ("days", "day"),
    "pdg_spearman": (),
    "ovulation_day": (),
    "cycle_regularity": (),
    "top_feature": (),
}
# Slots whose rendered value already starts with a word the model tends to write ahead of it.
_REDUNDANT_BEFORE = {"ovulation_day": ("day",)}


def _dedupe_units(text: str, values: dict[str, str]) -> str:
    """Remove a unit the model wrote next to a slot value that already carries it.

    Deliberately narrow: a phrase is only removed when it sits immediately beside the exact rendered
    value, so ordinary prose that happens to contain "days" is untouched.
    """
    for slot, phrases in _REDUNDANT_AFTER.items():
        val = values.get(slot)
        if not val:
            continue
        for phrase in phrases:  # longest first, so "degrees celsius" wins over "degrees"
            text = re.sub(
                rf"({re.escape(val)})\s+{re.escape(phrase)}\b",
                r"\1",
                text,
                flags=re.IGNORECASE,
            )
    for slot, words in _REDUNDANT_BEFORE.items():
        val = values.get(slot)
        if not val:
            continue
        for word in words:
            text = re.sub(
                rf"\b{re.escape(word)}\s+({re.escape(val)})",
                r"\1",
                text,
                flags=re.IGNORECASE,
            )
    return text


def _slot_values(p: ExplainPayload) -> dict[str, str]:
    regularity = p.cycle_regularity if p.cycle_regularity in _ALLOWED_REGULARITY else "unknown"
    feature = p.top_feature if _SAFE_FEATURE_RE.match(p.top_feature or "") else _DEFAULT_FEATURE
    # A caller cannot smuggle digits in through the feature name either.
    if any(ch.isdigit() for ch in feature):
        feature = _DEFAULT_FEATURE

    # A sentinel must never reach prose. -1 means "no ovulation day was decided", and rendering that
    # as "day -1" reads as a measurement rather than an absence.
    ov = f"day {p.model_ovulation_day}" if p.model_ovulation_day >= 0 else "no confidently identified day"

    return {
        "rhr_delta": f"{p.rhr_delta_bpm:+.1f} bpm",
        "temp_delta": f"{p.temp_delta_c:+.2f} °C",
        "pdg_spearman": f"{p.pdg_spearman:.2f}",
        "ovulation_day": ov,
        "cycle_regularity": regularity,
        "calendar_mae": f"{p.calendar_mae_days:.1f} days",
        "model_mae": f"{p.model_mae_days:.1f} days",
        "top_feature": feature,
    }


def _template_explanation(p: ExplainPayload) -> Explanation:
    """Deterministic explanation — the default and demo path. No model call."""
    tmpl = (
        "For this {{cycle_regularity}} cycle, the model places ovulation at {{ovulation_day}}. "
        "It leans most on {{top_feature}}: the wearable shows a temperature shift of {{temp_delta}} "
        "and a resting-heart-rate change of {{rhr_delta}} in the days after ovulation — the "
        "post-ovulatory progesterone signature [temp_luteal][rhr_luteal]. Across this participant, "
        "the model's daily PdG estimate tracks the urine assay at a rank correlation of "
        "{{pdg_spearman}}. On ovulation timing it is off by {{model_mae}} versus {{calendar_mae}} "
        "for the calendar — the calendar assumes a fixed cycle length and struggles most here "
        "[calendar_fails_irregular]. Because temperature rises only after ovulation, none of this "
        "can forecast ovulation in advance [temp_lags_ovulation]. This is a physiological estimate, "
        "not a diagnosis, and not contraceptive guidance [not_diagnostic]."
    )
    citations = ["temp_luteal", "rhr_luteal", "calendar_fails_irregular", "temp_lags_ovulation", "not_diagnostic"]
    # strip citation markers for slot check, then render
    slots = guard.SLOT_RE.findall(tmpl)
    vals = _slot_values(p)
    text = _dedupe_units(guard.render(_strip_citations(tmpl), vals), vals)
    return Explanation(
        text=text,
        citations=citations,
        slots_used=slots,
        grounded=True,
        source="template",
    )


def _strip_citations(t: str) -> str:
    import re

    return re.sub(r"\s*\[[a-z_]+\]", "", t)


def explain(p: ExplainPayload, question: str | None = None, use_llm: bool | None = None) -> Explanation:
    """Produce a grounded explanation. Refuses disallowed questions before any model call."""
    if question:
        # The deep gate, not the regex alone: a Cyrillic homoglyph in a condition name got this
        # endpoint to answer a diagnosis question with a full clinical narrative.
        cat = guard.classify_refusal_deep(question)
        if cat:
            return Explanation(
                text=guard.REFUSAL_COPY[cat],
                citations=["not_diagnostic"],
                slots_used=[],
                grounded=True,
                source="refusal",
            )

    want_llm = use_llm if use_llm is not None else bool(os.environ.get("OPENAI_API_KEY"))
    if not want_llm:
        return _template_explanation(p)

    try:
        return _llm_explanation(p, question)
    except Exception as e:  # noqa: BLE001 — any failure degrades to the deterministic template
        exp = _template_explanation(p)
        exp.warnings.append(f"llm path failed, used template: {type(e).__name__}")
        return exp


def _llm_explanation(p: ExplainPayload, question: str | None) -> Explanation:
    """Model-backed path.

    Uses the stdlib OpenAI client rather than the official SDK so that this exact function runs
    unchanged inside the 500MB Vercel serverless bundle as well as in the Docker backend.
    """
    from infradian.aio import openai_client as oai

    system = (PROMPT_DIR / "explain.system.v1.md").read_text()
    prompt_hash = hashlib.sha256(system.encode()).hexdigest()[:12]
    # Show the model what each slot RENDERS TO, not just its name. Given only names it assumes the
    # slot is a bare number and writes the unit itself, which reads as "+2.5 bpm beats per minute".
    vals = _slot_values(p)
    slot_list = "\n".join(
        f"  {{{{{k}}}}} renders to \"{vals[k]}\"" for k in sorted(guard.KNOWN_SLOTS)
    )
    # Interpolate the VALIDATED value, never the raw payload field. The allow-list was applied only
    # at render time, so arbitrary caller text still reached the prompt and came back as
    # model-authored clinical prose naming conditions, which the slot and digit checks cannot catch
    # because they inspect the template rather than the values.
    user = (
        f"Cycle regularity: {vals['cycle_regularity']}. You MUST use these slots for every number, and "
        f"each one ALREADY CONTAINS ITS UNIT, so never write the unit yourself:\n{slot_list}\n"
        f"Available evidence tags: {', '.join(evidence.EVIDENCE_BY_ID)}.\n"
        f"Write a 4–6 sentence explanation using ONLY {{{{slot}}}} placeholders for numbers and "
        f"[evidence_tag] markers for claims. Do NOT write any digits.\n"
    )
    # The user's question is deliberately NOT interpolated into the prompt. Output validation only
    # checks slot names and digits, so a digit-free medical sentence would render verbatim under a
    # "grounded" badge. Disallowed questions are refused upstream; allowed ones do not change what
    # the explanation should say, because it is fully determined by the payload.

    # JSON-constrained so the slot template comes back in a single predictable field.
    user += '\nReturn JSON: {"explanation": "<the slot-and-tag prose>"}'

    obj = oai.chat_json(
        system + "\n\nEVIDENCE:\n" + evidence.as_context(),
        user,
        model=os.environ.get("INFRADIAN_LLM_MODEL") or None,
        temperature=0.2,
        max_tokens=450,
    )
    template = str(obj.get("explanation", "")).strip()
    if not template:
        raise ValueError("model returned no explanation field")

    unknown = guard.verify_slots_known(template)
    free_nums = guard.verify_no_free_numbers(template)
    if unknown or free_nums:
        # The model broke the contract; fail closed to the deterministic template.
        exp = _template_explanation(p)
        # Client-facing wording only. The previous message exposed internal variable names and
        # the raw validator lists, which is debug output, not something a user can act on.
        exp.warnings.append("model output failed the grounding check, used the deterministic template")
        return exp

    import re

    citations = re.findall(r"\[([a-z_]+)\]", template)
    slots = guard.SLOT_RE.findall(template)
    vals = _slot_values(p)
    text = _dedupe_units(guard.render(_strip_citations(template), vals), vals)
    return Explanation(
        text=text,
        citations=[c for c in citations if c in evidence.EVIDENCE_BY_ID],
        slots_used=slots,
        grounded=True,
        source="llm",
        prompt_hash=prompt_hash,
    )
