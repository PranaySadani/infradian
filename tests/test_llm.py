"""LLM explanation-layer guarantees: numeric grounding by construction and refusal routing."""

from __future__ import annotations

from infradian.llm import evidence, guard
from infradian.llm.eval import BENIGN, REDTEAM, SAMPLE
from infradian.llm.explain import _template_explanation, explain


def test_template_has_only_known_slots():
    exp = _template_explanation(SAMPLE)
    assert all(s in guard.KNOWN_SLOTS for s in exp.slots_used)


def test_all_citations_resolve_to_evidence():
    exp = _template_explanation(SAMPLE)
    assert all(c in evidence.EVIDENCE_BY_ID for c in exp.citations)


def test_free_number_detector_flags_stray_digits():
    assert guard.verify_no_free_numbers("the rise was 5 bpm") == ["5"]
    assert guard.verify_no_free_numbers("the rise was {{rhr_delta}}") == []


def test_unknown_slot_detector():
    assert guard.verify_slots_known("{{made_up_slot}}") == ["made_up_slot"]
    assert guard.verify_slots_known("{{rhr_delta}}") == []


def test_redteam_prompts_are_refused():
    for q, _cat in REDTEAM:
        exp = explain(SAMPLE, question=q, use_llm=False)
        assert exp.source == "refusal", f"failed to refuse: {q!r}"


def test_benign_prompts_are_not_refused():
    for q in BENIGN:
        exp = explain(SAMPLE, question=q, use_llm=False)
        assert exp.source != "refusal", f"wrongly refused benign question: {q!r}"


def test_render_fails_closed_on_missing_slot():
    import pytest

    with pytest.raises(KeyError):
        guard.render("{{unknown_value}}", {})
