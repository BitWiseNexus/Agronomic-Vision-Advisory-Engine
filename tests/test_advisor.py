"""Guard rails around plan synthesis.

The behaviour under test is the safety-critical part of the project: a plan must never
carry a pesticide dosage that no retrieved source supports, and the measured severity tier
must win over whatever urgency the model asserts.
"""

from __future__ import annotations

import pytest

from app.llm.advisor import build_user_prompt, generate_plan, parse_plan, template_plan
from app.llm.provider import LLMProvider, LLMUnavailable, NullProvider, extract_json_object
from app.schemas.diagnosis import (
    ActionTier,
    ClassPrediction,
    Diagnosis,
    Evidence,
    SeverityReport,
)
from app.schemas.environment import EnvironmentContext, SoilType


def make_diagnosis(*, trained: bool = True, confidence: float = 0.91) -> Diagnosis:
    return Diagnosis(
        top=ClassPrediction(
            label="Tomato___Late_blight",
            crop="Tomato",
            disease="Late blight",
            confidence=confidence,
        ),
        alternatives=[],
        checkpoint_loaded=trained,
        confidence_reliable=trained and confidence >= 0.45,
        is_healthy=False,
    )


def make_severity(severity_pct: float) -> SeverityReport:
    from app.vision.severity import classify_tier

    return SeverityReport(
        severity_pct=severity_pct,
        necrotic_pct=severity_pct * 0.6,
        chlorotic_pct=severity_pct * 0.4,
        healthy_pct=100 - severity_pct,
        lesion_count=6,
        largest_lesion_pct=severity_pct / 3,
        leaf_coverage_pct=40.0,
        action_tier=classify_tier(severity_pct),
    )


EVIDENCE = [Evidence(id="S1", text="Mancozeb 75 % WP at 2 g per litre.", source="label", score=0.8)]


class StubProvider(LLMProvider):
    name = "stub"

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, *, json_mode: bool = True) -> str:
        self.calls += 1
        return self.payload


# --- prompt construction ----------------------------------------------------


def test_prompt_states_the_tier_mandate_and_severity() -> None:
    severity = make_severity(35.0)
    prompt = build_user_prompt(make_diagnosis(), severity, EnvironmentContext(), EVIDENCE)

    assert "35.0 %" in prompt
    assert "SEVERE" in prompt
    assert severity.action_tier.mandate in prompt
    assert "[S1]" in prompt


def test_prompt_marks_an_untrained_diagnosis_provisional() -> None:
    prompt = build_user_prompt(
        make_diagnosis(trained=False), make_severity(12.0), EnvironmentContext(), EVIDENCE
    )
    assert "PROVISIONAL" in prompt


def test_prompt_forbids_chemicals_when_no_evidence_was_retrieved() -> None:
    prompt = build_user_prompt(make_diagnosis(), make_severity(30.0), EnvironmentContext(), [])
    assert "Do NOT name any chemical active" in prompt


def test_prompt_includes_local_context() -> None:
    environment = EnvironmentContext(
        region="Nashik", soil_type=SoilType.BLACK, rainfall_last_7d_mm=140
    )
    prompt = build_user_prompt(make_diagnosis(), make_severity(20.0), environment, EVIDENCE)

    assert "Nashik" in prompt
    assert "black" in prompt
    assert "140 mm" in prompt
    assert "wash off" in prompt  # the heavy-rainfall interpretation


# --- the safety filter ------------------------------------------------------


def test_uncited_dosage_is_stripped() -> None:
    payload = {
        "summary": "Late blight, severe.",
        "urgency": "severe",
        "chemical_options": [
            "Mancozeb 75 % WP at 2 g per litre [S1]",
            "Chlorothalonil at 3 ml per litre",
        ],
    }
    plan = parse_plan(payload, make_severity(35.0), EVIDENCE, EnvironmentContext(), "stub")

    assert plan.unsupported_claims_removed == 1
    assert len(plan.chemical_options) == 1
    assert "S1" in plan.chemical_options[0]
    assert any("supported" in caution for caution in plan.cautions)


def test_citation_pointing_at_a_nonexistent_source_is_stripped() -> None:
    payload = {"chemical_options": ["Copper oxychloride at 3 g per litre [S9]"]}
    plan = parse_plan(payload, make_severity(30.0), EVIDENCE, EnvironmentContext(), "stub")

    assert plan.chemical_options == []
    assert plan.unsupported_claims_removed == 1


def test_organic_only_removes_every_chemical_option() -> None:
    payload = {"chemical_options": ["Mancozeb 75 % WP at 2 g per litre [S1]"]}
    plan = parse_plan(
        payload, make_severity(35.0), EVIDENCE, EnvironmentContext(organic_only=True), "stub"
    )
    assert plan.chemical_options == []


def test_measured_tier_overrides_the_model_urgency() -> None:
    payload = {"summary": "Just monitor.", "urgency": "negligible"}
    plan = parse_plan(payload, make_severity(40.0), EVIDENCE, EnvironmentContext(), "stub")
    assert plan.urgency is ActionTier.SEVERE


# --- fallback behaviour -----------------------------------------------------


def test_generate_plan_falls_back_when_the_llm_is_unavailable() -> None:
    plan = generate_plan(
        make_diagnosis(),
        make_severity(30.0),
        EnvironmentContext(),
        EVIDENCE,
        NullProvider("no key"),
    )
    assert plan.generated_by == "template"
    assert plan.chemical_options == []  # the template never names a product
    assert plan.immediate_actions


def test_generate_plan_falls_back_on_unparseable_output() -> None:
    plan = generate_plan(
        make_diagnosis(),
        make_severity(30.0),
        EnvironmentContext(),
        EVIDENCE,
        StubProvider("Sorry, I cannot help with that."),
    )
    assert plan.generated_by == "template"


def test_generate_plan_uses_a_valid_completion() -> None:
    completion = (
        '{"summary": "Severe late blight.", "urgency": "severe", '
        '"immediate_actions": [{"order": 1, "action": "Spray", "timing": "today", '
        '"citations": ["S1"]}], '
        '"chemical_options": ["Mancozeb 75 % WP at 2 g per litre [S1]"], '
        '"monitoring": "48h"}'
    )
    provider = StubProvider(completion)
    plan = generate_plan(
        make_diagnosis(), make_severity(35.0), EnvironmentContext(), EVIDENCE, provider
    )

    assert provider.calls == 1
    assert plan.generated_by == "stub"
    assert plan.summary == "Severe late blight."
    assert plan.chemical_options == ["Mancozeb 75 % WP at 2 g per litre [S1]"]


def test_template_plan_leads_with_confirmation_when_provisional() -> None:
    plan = template_plan(
        make_diagnosis(trained=False), make_severity(30.0), EnvironmentContext(), EVIDENCE, "no key"
    )
    assert plan.immediate_actions[0].order == 1
    assert "Confirm the identification" in plan.immediate_actions[0].action
    assert [step.order for step in plan.immediate_actions] == list(
        range(1, len(plan.immediate_actions) + 1)
    )


def test_template_plan_for_a_healthy_leaf_recommends_no_treatment() -> None:
    plan = template_plan(make_diagnosis(), make_severity(0.5), EnvironmentContext(), [], "no key")
    assert plan.urgency is ActionTier.NEGLIGIBLE
    assert plan.chemical_options == []
    assert "baseline" in " ".join(step.action for step in plan.immediate_actions).lower()


# --- provider plumbing ------------------------------------------------------


def test_null_provider_raises_llm_unavailable() -> None:
    with pytest.raises(LLMUnavailable):
        NullProvider("no key").generate("s", "u")


@pytest.mark.parametrize(
    "raw",
    [
        '{"summary": "x"}',
        '```json\n{"summary": "x"}\n```',
        'Here you go:\n{"summary": "x"}\nHope that helps!',
    ],
)
def test_json_extraction_tolerates_fences_and_prose(raw: str) -> None:
    assert extract_json_object(raw)["summary"] == "x"


def test_json_extraction_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        extract_json_object("no object here")
