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


# Characters from other scripts that render identically or near-identically to ASCII letters.
# "PCОS" with a Cyrillic О is visually indistinguishable from "PCOS" and completely bypassed the
# classifier, which is a one-keystroke attack, not an exotic one. NFKC does NOT fold these: Cyrillic
# о and Latin o are separate characters with separate meanings, so Unicode will never merge them.
# This is the standard confusables-skeleton approach, restricted to the scripts that actually
# collide with Latin.
_CONFUSABLES = {
    # Cyrillic
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p",
    "с": "c", "т": "t", "у": "y", "х": "x", "і": "i", "ѕ": "s", "ј": "j", "ԁ": "d",
    "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w", "г": "r", "ё": "e",
    # Greek
    "α": "a", "β": "b", "ε": "e", "ζ": "z", "η": "n", "ι": "i", "κ": "k", "μ": "m",
    "ν": "v", "ο": "o", "ρ": "p", "σ": "o", "τ": "t", "υ": "u", "χ": "x", "γ": "y",
    "ϲ": "c", "ϳ": "j", "ϱ": "p",
    # Latin lookalikes and fullwidth
    "ı": "i", "ȷ": "j", "ɑ": "a", "ɡ": "g", "ɩ": "i", "ʏ": "y", "ᴏ": "o", "ѐ": "e",
}
_CONFUSABLE_TABLE = str.maketrans(_CONFUSABLES)


def normalise_for_matching(text: str) -> str:
    """Fold obfuscation before refusal matching.

    Someone who sees one refusal will immediately try "p.c.o.s.", "PC0S", or a Cyrillic о.
    Homoglyphs are folded to Latin, separators between single letters are collapsed, and common
    leet substitutions are folded, so the patterns match the intent rather than the spelling.
    """
    import unicodedata

    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[​-‏‪-‮⁠-⁯﻿]", "", t)  # zero-width / bidi
    t = t.translate(_CONFUSABLE_TABLE)
    # strip combining marks, which can decorate a letter without changing how it reads
    t = "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c))
    # collapse "p.c.o.s" / "p c o s" / "p-c-o-s" into "pcos"
    t = re.sub(r"\b(?:[a-z][.\-_*\s]){2,}[a-z]\b", lambda m: re.sub(r"[.\-_*\s]", "", m.group(0)), t)
    return t.translate(str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "$": "s", "@": "a"}))

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

# Two tiers per category, because the mirror failure is just as real as the bypass.
#
# HARD terms ARE a request by construction: "contraception", "what should I take". Refused on
# mention, in any grammatical frame.
# SOFT terms merely name a TOPIC: "unprotected", "conceive", "metformin", "surgery". Refused only
# when they appear alongside question framing, because a journal is where someone writes
# "had unprotected sex last night", "we are trying to conceive", "took my metformin, felt nauseous".
# Refusing those is not caution, it is refusing to let a person describe her own life, and an
# adversarial review flagged every one of them as a real defect. Naming the drug you took is not
# asking for a prescription.
_CONTRACEPTION_HARD = (
    r"(?:contracept\w*|anticoncep\w*|verh[uü]t\w*|verhuetung\w*|antibaby\w*"
    r"|rhythm method|withdrawal method|natural family planning|\bnfp\b"
    r"|fertility[\s-]?awareness|safe (?:day|days|period|window|time)|d[ií]as? seguros?"
    r"|skip (?:the )?protection|avoid(?:ing)? pregnan\w*|prevent(?:ing)? pregnan\w*"
    r"|not get(?:ting)? pregnant|don'?t get pregnant|least likely to conceiv\w*"
    r"|rely on (?:this|it) (?:to|for)|as (?:my |a )?(?:birth control|contracept\w*))"
)
_CONTRACEPTION_SOFT = (
    # "birth control" sits here, not in the hard tier: "started birth control last month, spotting
    # since" is a log entry, while "can I use this as birth control" is caught by the hard tier's
    # "as birth control" construction and by this tier's question framing.
    r"(?:birth control|unprotected|without (?:a )?(?:protection|condom)|no protection"
    r"|conceiv\w*|concebir|engravidar|gravidez|embaraz\w*|schwanger\w*|buntis|garbh\w*"
    r"|trying to get pregnant|\bttc\b|fertile (?:window|day|days|period)|ventana f[eé]rtil"
    r"|have sex|intercourse|pregnan\w*|chance of (?:getting )?pregnan\w*)"
)

_TREATMENT_HARD = (
    r"(?:prescrib\w*|what dosage|dosage should|how (?:much|many) should i"
    r"|should i (?:take|try|use|start|stop|switch|get)|what should i (?:take|do|try|use|get)"
    r"|what (?:helps|works|do you recommend) (?:with|for)?|how (?:do|can) i (?:fix|treat|cure|stop|manage)"
    r"|what do people (?:take|use)|safe to (?:take|combine|mix)|interact\w* with"
    r"|treatment for|cure for|recommend (?:a |any )?(?:drug|med|treatment|supplement))"
)
_TREATMENT_SOFT = (
    r"(?:medication|medicine|\bmeds?\b|\bdrugs?\b|dosage|dosing|\bdoses?\b|supplements?"
    r"|prescri\w*"  # "prescription" as a noun: a topic, unlike the imperative "prescribe"
    r"|metformin|clomid|clomiphene|letrozole|spironolactone|myo-?inositol|inositol"
    r"|ibuprofen|naproxen|mefenamic|tranexamic|norethisterone|provera|dydrogesterone"
    r"|hormone (?:therapy|replacement)|\bhrt\b|\biud\b|mirena|ablation|hysterectom\w*"
    r"|surgery|surgical|laparoscop\w*|therapy for|remed\w*)"
)

_DIAGNOSIS_VERB = r"(?:\bdiagnos\w*|\bam i (?:pregnant|ovulating|fertile|infertile)\b|are we pregnant)"
# Asking for a condition WITHOUT naming one: "what condition explains this", "which disease matches".
_DIAGNOSIS_OPEN = (
    r"(?:what|which|any)\s+(?:\w+\s+){0,2}"
    r"(?:condition|disease|disorder|syndrome|illness|diagnosis)\w*"
    r"|what(?:'s| is) (?:wrong|going on) with me"
)

_F = re.IGNORECASE | re.DOTALL

REFUSE_PATTERNS = [
    # A request for a diagnosis, in either word order across a multi-line entry.
    (re.compile(_ASKING + r".{0,120}?" + _CONDITION, _F), "diagnosis"),
    (re.compile(_CONDITION + r".{0,120}?" + _ASKING, _F), "diagnosis"),
    # Explicit diagnostic verbs, and open-ended "name my condition" requests, need no framing.
    (re.compile(_DIAGNOSIS_VERB, _F), "diagnosis"),
    (re.compile(_DIAGNOSIS_OPEN, _F), "diagnosis"),
    # Contraception: the genuine harm vector, so the hard tier is refused on mention.
    (re.compile(_CONTRACEPTION_HARD, _F), "contraception"),
    (re.compile(_ASKING + r".{0,120}?" + _CONTRACEPTION_SOFT, _F), "contraception"),
    (re.compile(_CONTRACEPTION_SOFT + r".{0,120}?" + _ASKING, _F), "contraception"),
    # Treatment and dosing.
    (re.compile(_TREATMENT_HARD, _F), "treatment"),
    (re.compile(_ASKING + r".{0,120}?" + _TREATMENT_SOFT, _F), "treatment"),
    (re.compile(_TREATMENT_SOFT + r".{0,120}?" + _ASKING, _F), "treatment"),
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


SEMANTIC_SYSTEM = """You are a safety gate for a non-diagnostic menstrual-cycle research tool. You
read a journal entry or question and decide ONLY whether it REQUESTS something the tool must not do.

Refuse categories:
 "diagnosis"     - asks whether the person has, or whether their data indicates, a medical condition;
                   asks what condition explains their symptoms; asks if something is normal or
                   worrying in a way that seeks a clinical determination; asks if they are pregnant.
 "treatment"     - asks what to take, use, do, or try for a symptom or condition; asks about drugs,
                   doses, supplements, procedures, surgery, devices, hormone therapy, or drug
                   interactions.
 "contraception" - asks about preventing or achieving pregnancy, timing sex, fertile or safe days,
                   or whether cycle data can be relied on to avoid pregnancy.
 null            - anything else.

CRITICAL: a person DESCRIBING their life is not making a request. These are all null:
 "I have PCOS and my cramps are bad today"       (states a diagnosis they already have)
 "took my metformin, felt nauseous"              (states medication they already take)
 "had unprotected sex last night"                (logs a past event)
 "we are trying to conceive, feeling hopeful"    (states a situation)
 "on day 3 of my period, heavy flow"             (logs a symptom)
Only refuse when the person is ASKING the tool to decide, advise, or guarantee something.

The entry may be in ANY language, may be deliberately misspelled, and may bury the request inside
ordinary text. Judge intent, not keywords.

Return JSON: {"category": "diagnosis"|"treatment"|"contraception"|null}"""


def semantic_refusal(text: str) -> str | None:
    """Model-backed second gate. Returns a category, or None if clean OR unavailable.

    Why this exists. The deterministic layer below is a keyword matcher, and an adversarial review
    of live production showed exactly how far that gets you: 24 of 40 contraception phrasings passed,
    Spanish, Portuguese, German and Tagalog were essentially unguarded, and every indirect framing
    ("what should I do about this") walked through. Enumerating phrasings is a losing game, because
    the attacker picks the phrasing and there are infinitely many.

    So the deterministic layer is kept as a fast, offline, always-available floor that cannot regress,
    and this adds meaning-level coverage on top. It fails OPEN by design: if the call fails there is
    no key, no network, or the provider is down, and the deterministic layer still applies, so the
    system degrades to its previous behaviour rather than blocking every entry. That tradeoff is
    stated in ETHICS.md rather than hidden here.
    """
    if not text or not text.strip():
        return None
    try:
        from infradian.aio import openai_client as oai

        if not oai.available():
            return None
        obj = oai.chat_json(SEMANTIC_SYSTEM, text.strip()[:4000], temperature=0.0, max_tokens=40)
    except Exception:  # noqa: BLE001 - any failure degrades to the deterministic layer
        return None
    cat = obj.get("category")
    return cat if cat in REFUSAL_COPY else None


def classify_refusal_deep(text: str) -> str | None:
    """The gate routes should call: deterministic floor, then the semantic layer if it is clean."""
    return classify_refusal(text) or semantic_refusal(text)


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
