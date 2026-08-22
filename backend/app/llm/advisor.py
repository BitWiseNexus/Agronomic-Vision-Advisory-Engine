"""Prompt assembly and plan synthesis — where the severity index becomes a constraint.

Three things make this more than "ask an LLM about a plant":

1. The measured action tier is stated as a **mandate** the model must obey, so a 3 %
   lesion ratio cannot produce a fungicide recommendation.
2. Retrieved chunks are numbered evidence and the model must cite by id; anything it
   cannot support it must label unsupported.
3. `_strip_uncited_chemicals` deletes any chemical recommendation that carries no
   citation. Dosage hallucination is the one failure mode in this project that could
   actually harm a crop or a person, so it is handled in code, not by asking nicely.
"""

from __future__ import annotations

import re

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.llm.provider import (
    LLMProvider,
    LLMUnavailable,
    extract_json_object,
    resolve_provider,
)
from app.schemas.diagnosis import (
    ActionTier,
    Diagnosis,
    Evidence,
    PlanStep,
    SeverityReport,
    TreatmentPlan,
)
from app.schemas.environment import EnvironmentContext

log = get_logger(__name__)

SYSTEM_PROMPT = """\
You are an agricultural extension advisor. You write treatment plans for smallholder \
farmers based on a photograph analysed by computer vision plus field conditions the \
farmer reported.

Non-negotiable rules:
1. The MEASURED SEVERITY and its ACTION TIER are ground truth from pixel measurement. \
Never contradict them and never escalate or de-escalate beyond the tier mandate you are \
given.
2. Recommend a chemical active ONLY if a numbered evidence snippet supports it. Put the \
snippet id in that step's "citations". Never invent a dose, concentration, spray \
interval or pre-harvest interval that is not in the evidence.
3. If the evidence does not cover something, say so in "cautions" rather than guessing.
4. If the diagnosis is marked PROVISIONAL, hedge the plan and make confirming the \
identification the first action.
5. Tailor timing and product choice to the reported local conditions (soil, rainfall, \
season, irrigation). Rainfall matters: contact sprays wash off.
6. Plain, practical language. Every action must be something a farmer can actually do.

Reply with ONLY a JSON object of this exact shape:
{
  "summary": "2-3 sentences: what it is, how bad, what happens next",
  "urgency": "negligible|early|moderate|severe|critical",
  "immediate_actions": [
    {"order": 1, "action": "...", "timing": "...", "rationale": "...", "citations": ["S1"]}
  ],
  "followup_actions": [
    {"order": 1, "action": "...", "timing": "...", "rationale": "...", "citations": []}
  ],
  "chemical_options": ["active ingredient + rate + interval [S2]"],
  "cultural_practices": ["..."],
  "monitoring": "what to re-check, and when",
  "cautions": ["..."]
}
"""

_NUMERIC_RE = re.compile(r"\d")
_CITATION_RE = re.compile(r"\bS\d+\b")


def build_user_prompt(
    diagnosis: Diagnosis,
    severity: SeverityReport,
    environment: EnvironmentContext,
    evidence: list[Evidence],
) -> str:
    provisional = not (diagnosis.checkpoint_loaded and diagnosis.confidence_reliable)
    reason = (
        "no trained checkpoint is loaded, so this label is a PLACEHOLDER"
        if not diagnosis.checkpoint_loaded
        else "confidence is below the reliability floor"
    )

    lines: list[str] = ["## VISUAL DIAGNOSIS (PyTorch ResNet-50)"]
    lines.append(f"- Crop: {diagnosis.top.crop}")
    lines.append(f"- Suspected condition: {diagnosis.top.disease}")
    lines.append(f"- Model confidence: {diagnosis.top.confidence:.1%}")
    if provisional:
        lines.append(f"- STATUS: **PROVISIONAL** — {reason}. Treat identification as unconfirmed.")
    else:
        lines.append("- STATUS: confirmed by a trained classifier")
    if diagnosis.alternatives:
        alts = ", ".join(
            f"{alt.disease} ({alt.confidence:.0%})" for alt in diagnosis.alternatives[:2]
        )
        lines.append(f"- Other candidates: {alts}")

    lines.append("\n## MEASURED SEVERITY (OpenCV pixel analysis — ground truth)")
    lines.append(severity.to_prompt_block())
    if severity.quality_warnings:
        lines.append("- Image quality caveats: " + "; ".join(severity.quality_warnings))

    lines.append(f"\n## TIER MANDATE (you MUST comply)\n{severity.action_tier.mandate}")

    lines.append("\n## LOCAL CONDITIONS REPORTED BY THE FARMER")
    lines.extend(environment.to_prompt_lines())
    wetness = environment.wetness_note()
    if wetness:
        lines.append(f"- Interpretation: {wetness}")

    lines.append("\n## EVIDENCE (cite by id; nothing outside this is citable)")
    if evidence:
        for item in evidence:
            header = f"[{item.id}] {item.source}"
            if item.crop and item.crop != "general":
                header += f" — {item.crop}/{item.disease}"
            lines.append(f"{header}\n{item.text[:1600]}\n")
    else:
        lines.append(
            "(No documents retrieved. You may give only generic, tier-appropriate cultural "
            "advice. Do NOT name any chemical active or dose. Say the knowledge base had no "
            "matching material.)"
        )

    lines.append(
        "\n## TASK\nProduce the JSON treatment plan. Keep it consistent with the measured "
        f"severity of {severity.severity_pct:.1f} % and the "
        f"{severity.action_tier.value.upper()} tier mandate."
    )
    return "\n".join(lines)


def _coerce_steps(raw: object) -> list[PlanStep]:
    steps: list[PlanStep] = []
    if not isinstance(raw, list):
        return steps
    for position, entry in enumerate(raw, start=1):
        if isinstance(entry, str):
            steps.append(PlanStep(order=position, action=entry, timing="as soon as practical"))
            continue
        if not isinstance(entry, dict):
            continue
        action = str(entry.get("action") or entry.get("step") or "").strip()
        if not action:
            continue
        citations = entry.get("citations") or []
        if isinstance(citations, str):
            citations = _CITATION_RE.findall(citations)
        steps.append(
            PlanStep(
                order=int(entry.get("order") or position),
                action=action,
                timing=str(entry.get("timing") or "as soon as practical"),
                rationale=str(entry.get("rationale") or ""),
                citations=[str(item) for item in citations if isinstance(item, (str, int))],
            )
        )
    return steps


def _strip_uncited_chemicals(
    options: list[str], valid_ids: set[str], organic_only: bool
) -> tuple[list[str], int]:
    """A chemical line that names a number (a rate, a concentration, an interval) but
    cites nothing is a hallucination risk we refuse to pass to a farmer."""
    kept: list[str] = []
    removed = 0
    for option in options:
        text = str(option).strip()
        if not text:
            continue
        cited = {citation for citation in _CITATION_RE.findall(text) if citation in valid_ids}
        if organic_only or (_NUMERIC_RE.search(text) and not cited):
            removed += 1
            log.warning("Dropped uncited/disallowed chemical recommendation: %s", text[:120])
            continue
        kept.append(text)
    return kept, removed


def _as_str_list(raw: object, limit: int = 12) -> list[str]:
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()][:limit]


def parse_plan(
    payload: dict,
    severity: SeverityReport,
    evidence: list[Evidence],
    environment: EnvironmentContext,
    provider_name: str,
) -> TreatmentPlan:
    valid_ids = {item.id for item in evidence}
    chemicals, removed = _strip_uncited_chemicals(
        _as_str_list(payload.get("chemical_options")), valid_ids, environment.organic_only
    )

    # The measured tier wins over whatever urgency the model felt like reporting.
    try:
        stated = ActionTier(str(payload.get("urgency", "")).strip().lower())
    except ValueError:
        stated = severity.action_tier
    if stated is not severity.action_tier:
        log.info(
            "Model urgency '%s' overridden by measured tier '%s'",
            stated.value,
            severity.action_tier.value,
        )

    cautions = _as_str_list(payload.get("cautions"))
    if removed:
        cautions.append(
            f"{removed} chemical recommendation(s) were removed because they were not "
            "supported by a cited source."
        )
    needs_product = severity.action_tier is not ActionTier.NEGLIGIBLE
    if needs_product and not chemicals and not environment.organic_only:
        cautions.append(
            "No product recommendation is given here — confirm actives with your local "
            "extension office or the product label."
        )

    return TreatmentPlan(
        summary=str(payload.get("summary") or "").strip() or severity.action_tier.headline,
        urgency=severity.action_tier,
        immediate_actions=_coerce_steps(payload.get("immediate_actions")),
        followup_actions=_coerce_steps(payload.get("followup_actions")),
        chemical_options=chemicals,
        cultural_practices=_as_str_list(payload.get("cultural_practices")),
        monitoring=str(payload.get("monitoring") or "").strip(),
        cautions=cautions,
        generated_by=provider_name,
        unsupported_claims_removed=removed,
    )


# ---------------------------------------------------------------------------
# Deterministic fallback — no network, no keys, always available.
# ---------------------------------------------------------------------------

_TIER_IMMEDIATE: dict[ActionTier, list[tuple[str, str]]] = {
    ActionTier.NEGLIGIBLE: [
        ("Record this photo and date as a healthy baseline for the block.", "today"),
        ("Continue the normal scouting round; no treatment is warranted.", "ongoing"),
    ],
    ActionTier.EARLY: [
        ("Remove and destroy affected leaves away from the field — do not compost in place.", "within 48 hours"),
        ("Scout 10 plants across the block and count how many show the same symptom.", "within 48 hours"),
        ("Stop overhead irrigation; water at the base to keep foliage dry.", "immediately"),
    ],
    ActionTier.MODERATE: [
        ("Remove affected tissue from the worst plants and clear crop debris.", "within 24 hours"),
        ("Spot-treat only the affected plants and a one-plant buffer around them.", "within 48 hours"),
        ("Improve airflow: widen spacing, remove lower canopy leaves, control weeds.", "this week"),
        ("Mark affected plants so spread can be measured at the next inspection.", "within 24 hours"),
    ],
    ActionTier.SEVERE: [
        ("Treat the whole affected block on a label-rate schedule.", "within 24 hours"),
        ("Remove and destroy the most heavily affected plants to cut the inoculum load.", "within 24 hours"),
        ("Isolate the block: clean tools and footwear before entering healthy areas.", "immediately"),
        ("Check drainage and stop any overhead irrigation.", "immediately"),
    ],
    ActionTier.CRITICAL: [
        ("Treat immediately and assess whether this block is economic to save.", "today"),
        ("Protect neighbouring blocks first — they are the salvageable value.", "today"),
        ("Destroy the worst plants; do not move plant material between blocks.", "today"),
        ("Contact your local extension officer with these photos before replanting.", "within 48 hours"),
    ],
}

_TIER_MONITORING: dict[ActionTier, str] = {
    ActionTier.NEGLIGIBLE: "Re-photograph the same leaf position in 10-14 days and compare severity.",
    ActionTier.EARLY: "Re-photograph in 5-7 days. If severity passes 10 %, move to targeted treatment.",
    ActionTier.MODERATE: "Re-assess in 3-5 days. If severity passes 25 %, escalate to a full block spray.",
    ActionTier.SEVERE: "Re-assess every 48 hours through the treatment schedule.",
    ActionTier.CRITICAL: "Daily inspection of surrounding blocks for the next 10 days.",
}


def template_plan(
    diagnosis: Diagnosis,
    severity: SeverityReport,
    environment: EnvironmentContext,
    evidence: list[Evidence],
    reason: str,
) -> TreatmentPlan:
    """Rule-based plan derived entirely from the measured tier. Deliberately names no
    chemical active: without an LLM to read the evidence, we will not guess a product."""
    tier = severity.action_tier

    immediate = [
        PlanStep(order=index, action=action, timing=timing, rationale=tier.mandate)
        for index, (action, timing) in enumerate(_TIER_IMMEDIATE[tier], start=1)
    ]
    if not (diagnosis.checkpoint_loaded and diagnosis.confidence_reliable):
        immediate.insert(
            0,
            PlanStep(
                order=1,  # the whole list is renumbered immediately below
                action=(
                    f"Confirm the identification of '{diagnosis.top.disease}' on "
                    f"{diagnosis.top.crop} with an extension officer before treating — the "
                    "automated diagnosis is provisional."
                ),
                timing="before any treatment",
                rationale="Classifier confidence is not reliable for this image.",
            ),
        )
        for position, step in enumerate(immediate, start=1):
            step.order = position

    cultural = [
        "Sanitise pruning tools between plants with a 10 % bleach or 70 % alcohol wipe.",
        "Remove volunteer plants and weeds that host the same pathogens.",
        "Rotate away from the same crop family in this block next season.",
    ]
    wetness = environment.wetness_note()
    if wetness:
        cultural.insert(0, wetness)
    if environment.irrigation.value in {"flood", "sprinkler"}:
        cultural.insert(
            0,
            f"Current {environment.irrigation.value} irrigation keeps foliage wet — "
            "switch to drip or base watering while symptoms are active.",
        )

    cautions = [
        "This plan was generated without a language model (deterministic fallback), so it "
        f"names no chemical product: {reason}",
        "Verify any product choice, dose and pre-harvest interval with the label and your "
        "local extension officer.",
    ]
    if severity.quality_warnings:
        cautions.extend(severity.quality_warnings)
    if not evidence:
        cautions.append("No knowledge-base documents matched, so this is generic tier guidance.")

    summary = (
        f"{diagnosis.top.crop}: suspected {diagnosis.top.disease.lower()} with a measured "
        f"{severity.severity_pct:.1f} % of leaf area affected "
        f"({severity.necrotic_pct:.1f} % dead tissue, {severity.lesion_count} distinct lesions). "
        f"That places this at the {tier.value.upper()} tier — {tier.headline.lower()}."
    )

    return TreatmentPlan(
        summary=summary,
        urgency=tier,
        immediate_actions=immediate,
        followup_actions=[
            PlanStep(
                order=1,
                action="Re-photograph the same leaves and re-run the analysis to measure trend.",
                timing=_TIER_MONITORING[tier],
                rationale="Severity change over time decides whether to escalate.",
            )
        ],
        chemical_options=[],
        cultural_practices=cultural,
        monitoring=_TIER_MONITORING[tier],
        cautions=cautions,
        generated_by="template",
        unsupported_claims_removed=0,
    )


# ---------------------------------------------------------------------------


def generate_plan(
    diagnosis: Diagnosis,
    severity: SeverityReport,
    environment: EnvironmentContext,
    evidence: list[Evidence],
    provider: LLMProvider | None = None,
    settings: Settings | None = None,
) -> TreatmentPlan:
    """Synthesise the plan. Never raises — degrades to the template plan."""
    settings = settings or get_settings()
    provider = provider or resolve_provider(settings)
    user_prompt = build_user_prompt(diagnosis, severity, environment, evidence)

    try:
        raw = provider.generate(SYSTEM_PROMPT, user_prompt, json_mode=True)
        payload = extract_json_object(raw)
        return parse_plan(payload, severity, evidence, environment, provider.name)
    except LLMUnavailable as exc:
        log.warning("LLM unavailable (%s) — using the deterministic template plan", exc)
        return template_plan(diagnosis, severity, environment, evidence, str(exc))
    except Exception as exc:
        log.error("Plan synthesis failed on %s output (%s) — using the template plan", provider.name, exc)
        return template_plan(
            diagnosis, severity, environment, evidence, f"{provider.name} returned unusable output: {exc}"
        )
