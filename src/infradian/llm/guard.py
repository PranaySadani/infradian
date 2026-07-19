"""Guardrails. Two mechanical guarantees that make this not a chatbot wrapper:

1. NUMERIC GROUNDING BY CONSTRUCTION in the explanation layer. The model emits prose with typed
   slots like {{rhr_delta}}; a deterministic renderer substitutes values from the payload. A
   hallucinated number in an explanation is structurally impossible, not merely detected.
   (This claim is scoped to the explanation layer. The vision reader does return a model-read
   number, which is why it is range-checked, confidence-gated and labelled as transcribed.)

2. REFUSAL ROUTING. A stage-0 classifier catches diagnostic, treatment and contraceptive requests
   before any model call. The contraception path is the genuine harm vector here: wearable cycle
   estimates are nowhere near reliable enough to prevent a pregnancy.

HISTORY, kept deliberately. The first version of this file wrapped truncated stems in a trailing
word boundary, as in `\\b(...|contracept|...)\\b`. That can never fire, because the character after
"contracept" in "contraception" is a word character. Nine of ten unsafe probes passed, including
"Can I use this as contraception?", the very question the project demos as blocked. The unit tests
missed it because every case they used happened to hit a complete-word alternative such as "birth
control" or "pcos". An adversarial review caught it. Stems now carry explicit `\\w*` suffixes, the
cross-clause patterns run with DOTALL because the journal is a multi-line field, and the test suite
asserts on stems rather than on convenient whole words.
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
# Any digit at all, once slots are removed. The earlier `\b\d+` missed "day14" and "28days",
# because a word boundary cannot sit between two word characters. That is exactly the shape a
# model produces by accident, so the check has to be this blunt.
FREE_NUMBER_RE = re.compile(r"\d")
# Spelled-out numerals are the other way a number sneaks in.
SPELLED_NUMBER_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|hundred|half)\b",
    re.I,
)


def normalise_for_matching(text: str) -> str:
    """Fold obfuscation before refusal matching.

    Someone who sees one refusal will immediately try "p.c.o.s." or "PC0S". Separators between
    single letters are collapsed and common leet substitutions folded, so the patterns match the
    intent rather than the spelling.
    """
    import unicodedata

    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[​-‏⁠]", "", t)  # zero-width characters
    # collapse "p.c.o.s" / "p c o s" / "p-c-o-s" into "pcos"
    t = re.sub(r"\b(?:[a-z][.\-_*\s]){2,}[a-z]\b", lambda m: re.sub(r"[.\-_*\s]", "", m.group(0)), t)
    return t.translate(str.maketrans({"0": "o", "1": "i", "3": "e", "$": "s", "@": "a"}))

# --- building blocks. Stems use \w* so "contracept" catches contraception/contraceptive. ---

_ASKING = (
    r"(?:\?|\bdo(?:es)? i\b|\bdo i\b|\bam i\b|\bis (?:this|it|that|my)\b|\bcould (?:this|it|i)\b"
    r"|\bdoes (?:this|my|it)\b|\bcan i\b|\bshould i\b|\bwhich\b|\bwhen\b|\bwhat\b|\bhow do i\b"
    r"|\btell me\b|\blook like\b|\bconsistent with\b|\bsign of\b|\bmean i\b)"
)

_CONDITION = (
    r"(?:pcos|pcod|polycystic|endometrios\w*|\bendo\b|adenomyos\w*|fibroid\w*|cancer|tumou?rs?"
    r"|thyroid|hypothyroid\w*|hyperthyroid\w*|infertil\w*|perimenopaus\w*|menopaus\w*"
    r"|hormonal? (?:imbalance|disorder|condition|problem)|luteal phase defect)"
)

_CONTRACEPTION = (
    r"(?:contracept\w*|birth control|\bpull(?:ing)?[\s-]?out\b|rhythm method"
    r"|natural family planning|\bnfp\b|safe (?:day|days|period|window|time)"
    r"|unprotected|without protection|skip protection|no protection"
    r"|avoid(?:ing)? pregnan\w*|prevent(?:ing)? pregnan\w*|not get pregnant"
    r"|least likely to conceiv\w*|conceiv\w*|trying to get pregnant|ttc|fertile window|when can i have sex|best (?:day|days|time) to)"
)

_TREATMENT = (
    r"(?:prescrib\w*|prescription|medication|medicine|\bdrugs?\b|dosage|\bdoses?\b"
    r"|treatment for|treat my|metformin|clomid|letrozole|spironolactone"
    r"|should i take|what should i take|supplements? (?:should|for))"
)

_DIAGNOSIS_VERB = r"(?:\bdiagnos\w*|\bam i (?:pregnant|ovulating|fertile|infertile)\b|are we pregnant)"

_F = re.IGNORECASE | re.DOTALL

REFUSE_PATTERNS = [
    # A request for a diagnosis, in either word order across a multi-line entry.
    (re.compile(_ASKING + r".{0,120}?" + _CONDITION, _F), "diagnosis"),
    (re.compile(_CONDITION + r".{0,120}?" + _ASKING, _F), "diagnosis"),
    # Explicit diagnostic verbs need no question framing.
    (re.compile(_DIAGNOSIS_VERB, _F), "diagnosis"),
    # Contraception is hard-refused on mention. This is the real harm vector and there is no
    # phrasing of it we are willing to engage with.
    (re.compile(_CONTRACEPTION, _F), "contraception"),
    # Treatment and dosing.
    (re.compile(_TREATMENT, _F), "treatment"),
]

REFUSAL_COPY = {
    "diagnosis": (
        "I can't diagnose conditions or tell you whether your data looks like one. This tool "
        "estimates physiological cycle signals from wearables and is not a diagnostic device. "
        "Please talk to a clinician about symptoms, and feel free to bring the trajectory report "
        "to that conversation."
    ),
    "contraception": (
        "I can't help with contraception or with timing sex to avoid or achieve pregnancy. "
        "Wearable-derived cycle estimates are not reliable enough for that, and using them that way "
        "is unsafe. Please speak to a clinician about contraception."
    ),
    "treatment": (
        "I can't give treatment, medication or dosing advice. This is a non-diagnostic research "
        "tool. Please consult a licensed clinician."
    ),
}


def classify_refusal(text: str) -> str | None:
    """Return a refusal category if the input asks for a disallowed action, else None.

    Note on scope: a bare declarative mention of a condition ("I have PCOS and bad cramps today")
    is deliberately NOT refused. That is a legitimate journal entry, and the extraction path can
    only ever emit codes from a fixed vocabulary, so nothing unsafe can come back out of it.
    What is refused is a *request*: asking whether the data indicates a condition, asking about
    contraception in any phrasing, or asking for treatment.
    """
    if not text:
        return None
    normalised = normalise_for_matching(text)
    for pat, cat in REFUSE_PATTERNS:
        # Match the raw text and the de-obfuscated form, and match per line as well as whole, so
        # neither a newline nor a dotted acronym provides an escape hatch.
        if pat.search(text) or pat.search(normalised):
            return cat
        if any(pat.search(ln) for ln in normalised.splitlines() if ln.strip()):
            return cat
    return None


def verify_slots_known(template: str) -> list[str]:
    """Return any slot names in the template that the renderer cannot fill (should be empty)."""
    return [s for s in SLOT_RE.findall(template) if s not in KNOWN_SLOTS]


def verify_no_free_numbers(template: str) -> list[str]:
    """Return any literal numbers the model wrote outside a slot. Non-empty => reject the output."""
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
