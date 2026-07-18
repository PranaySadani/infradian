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


def _slot_values(p: ExplainPayload) -> dict[str, str]:
    return {
        "rhr_delta": f"{p.rhr_delta_bpm:+.1f} bpm",
        "temp_delta": f"{p.temp_delta_c:+.2f} °C",
        "pdg_spearman": f"{p.pdg_spearman:.2f}",
        "ovulation_day": f"day {p.model_ovulation_day}",
        "cycle_regularity": p.cycle_regularity,
        "calendar_mae": f"{p.calendar_mae_days:.1f} days",
        "model_mae": f"{p.model_mae_days:.1f} days",
        "top_feature": p.top_feature,
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
    text = guard.render(_strip_citations(tmpl), _slot_values(p))
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
        cat = guard.classify_refusal(question)
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
    from openai import OpenAI

    system = (PROMPT_DIR / "explain.system.v1.md").read_text()
    prompt_hash = hashlib.sha256(system.encode()).hexdigest()[:12]
    slot_list = ", ".join(sorted(guard.KNOWN_SLOTS))
    user = (
        f"Cycle regularity: {p.cycle_regularity}. Available slots you MUST use for every number: "
        f"{slot_list}. Available evidence tags: {', '.join(evidence.EVIDENCE_BY_ID)}.\n"
        f"Write a 4–6 sentence explanation using ONLY {{{{slot}}}} placeholders for numbers and "
        f"[evidence_tag] markers for claims. Do NOT write any digits.\n"
    )
    if question:
        user += f"The user asked: {question!r}. Answer only if it is a non-diagnostic question about the estimate.\n"

    client = OpenAI()
    resp = client.chat.completions.create(
        model=os.environ.get("INFRADIAN_LLM_MODEL", MODEL),
        messages=[{"role": "system", "content": system + "\n\nEVIDENCE:\n" + evidence.as_context()},
                  {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=350,
    )
    template = resp.choices[0].message.content or ""

    unknown = guard.verify_slots_known(template)
    free_nums = guard.verify_no_free_numbers(template)
    if unknown or free_nums:
        # The model broke the contract; fail closed to the deterministic template.
        exp = _template_explanation(p)
        exp.warnings.append(f"llm violated grounding (unknown_slots={unknown}, free_numbers={free_nums})")
        return exp

    import re

    citations = re.findall(r"\[([a-z_]+)\]", template)
    slots = guard.SLOT_RE.findall(template)
    text = guard.render(_strip_citations(template), _slot_values(p))
    return Explanation(
        text=text,
        citations=[c for c in citations if c in evidence.EVIDENCE_BY_ID],
        slots_used=slots,
        grounded=True,
        source="llm",
        prompt_hash=prompt_hash,
    )
