"""Guardrails. Two mechanical guarantees that make this not a chatbot wrapper:

1. NUMERIC GROUNDING BY CONSTRUCTION. The LLM emits prose with typed slots like {{rhr_delta}}; a
   deterministic renderer substitutes values from the payload. So a hallucinated number is
   structurally impossible, not merely detected. `verify_no_free_numbers` additionally rejects any
   literal digit the model wrote outside a slot — belt and suspenders.

2. REFUSAL ROUTING. A stage-0 regex classifier catches diagnostic / contraceptive / treatment
   questions and routes them to a fixed refusal before any model call. The contraception misuse
   path is a genuine harm vector and is handled explicitly.
"""

from __future__ import annotations

import re

# Slots the renderer knows how to fill. The model may only reference these.
KNOWN_SLOTS = frozenset(
    {
        "rhr_delta",
        "temp_delta",
        "pdg_spearman",
        "ovulation_day",
        "cycle_regularity",
        "calendar_mae",
        "model_mae",
        "top_feature",
    }
)

SLOT_RE = re.compile(r"\{\{(\w+)\}\}")
# A "free number" is any digit not inside a slot. Percent signs / decimals included.
FREE_NUMBER_RE = re.compile(r"(?<!\{)\b\d+(?:\.\d+)?%?\b(?!\})")

# Disallowed-intent patterns (stage-0 refusal).
REFUSE_PATTERNS = [
    (re.compile(r"\b(do i have|diagnos|is this|could it be)\b.*\b(pcos|endometrios|cancer|disease|disorder)\b", re.I), "diagnosis"),
    (re.compile(r"\b(pcos|endometriosis|cancer|thyroid disease)\b", re.I), "diagnosis"),
    (re.compile(r"\b(safe to|can i|when).*(unprotected|contracept|pull out|avoid pregnan|not get pregnant)\b", re.I), "contraception"),
    (re.compile(r"\b(birth control|contracept|rhythm method|pull.?out)\b", re.I), "contraception"),
    (re.compile(r"\b(should i take|what dose|prescrib|medication|drug|treatment for)\b", re.I), "treatment"),
    (re.compile(r"\b(am i pregnant|are we pregnant|conceiv(e|ing) this)\b", re.I), "diagnosis"),
]

REFUSAL_COPY = {
    "diagnosis": "I can't diagnose conditions. This tool estimates physiological cycle signals from wearables; it is not a diagnostic device. Please talk to a clinician about symptoms — and you can bring the trajectory report to that conversation.",
    "contraception": "I can't provide contraception or fertility-avoidance guidance. Wearable-derived cycle estimates are not reliable enough to prevent or plan pregnancy, and using them that way is unsafe. Please consult a clinician about contraception.",
    "treatment": "I can't give treatment or medication advice. This is a non-diagnostic research tool. Please consult a licensed clinician.",
}


def classify_refusal(text: str) -> str | None:
    """Return a refusal category if the input asks for a disallowed action, else None."""
    for pat, cat in REFUSE_PATTERNS:
        if pat.search(text):
            return cat
    return None


def verify_slots_known(template: str) -> list[str]:
    """Return any slot names in the template that the renderer cannot fill (should be empty)."""
    return [s for s in SLOT_RE.findall(template) if s not in KNOWN_SLOTS]


def verify_no_free_numbers(template: str) -> list[str]:
    """Return any literal numbers the model wrote outside a slot. Non-empty => reject the output."""
    # Remove slots first, then scan for stray digits.
    stripped = SLOT_RE.sub("", template)
    return FREE_NUMBER_RE.findall(stripped)


def render(template: str, values: dict[str, str]) -> str:
    """Fill slots from `values`. A missing value is a hard error (fail closed)."""
    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(f"no value for slot {{{{{key}}}}}")
        return values[key]

    return SLOT_RE.sub(sub, template)
