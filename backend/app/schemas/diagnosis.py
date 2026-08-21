"""Output contract. The API, the notebook and the tests all speak these types."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.environment import EnvironmentContext


class ActionTier(str, Enum):
    """Severity buckets. These are the guard rails the LLM is not allowed to cross."""

    NEGLIGIBLE = "negligible"
    EARLY = "early"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

    @property
    def mandate(self) -> str:
        return _TIER_MANDATE[self]

    @property
    def headline(self) -> str:
        return _TIER_HEADLINE[self]


_TIER_MANDATE: dict[ActionTier, str] = {
    ActionTier.NEGLIGIBLE: (
        "No treatment is justified. Record a baseline and re-inspect in 7-10 days. "
        "Do NOT recommend any spray, chemical or biological."
    ),
    ActionTier.EARLY: (
        "Monitoring and cultural control ONLY: sanitation, removal of affected tissue, "
        "spacing/airflow, irrigation adjustment. Do NOT recommend a blanket chemical spray."
    ),
    ActionTier.MODERATE: (
        "Targeted intervention on affected plants: biologicals and spot treatment first. "
        "A chemical active is permitted only if a cited source supports it at this severity."
    ),
    ActionTier.SEVERE: (
        "Immediate full chemical intervention on a label-rate schedule across the affected "
        "block, plus cultural measures to stop further spread."
    ),
    ActionTier.CRITICAL: (
        "Immediate intervention AND a salvage decision: assess whether treating this block "
        "is economic versus removal/replanting and protecting neighbouring blocks."
    ),
}

_TIER_HEADLINE: dict[ActionTier, str] = {
    ActionTier.NEGLIGIBLE: "Healthy — baseline monitoring",
    ActionTier.EARLY: "Early stage — monitor and sanitise",
    ActionTier.MODERATE: "Moderate — targeted treatment",
    ActionTier.SEVERE: "Severe — immediate intervention",
    ActionTier.CRITICAL: "Critical — intervene and assess salvage",
}


class ClassPrediction(BaseModel):
    label: str
    crop: str
    disease: str
    confidence: float = Field(ge=0.0, le=1.0)


class Diagnosis(BaseModel):
    """What the CNN thinks it is looking at."""

    top: ClassPrediction
    alternatives: list[ClassPrediction] = Field(default_factory=list)
    checkpoint_loaded: bool = Field(
        description="False => untrained head, the label is a placeholder, not a prediction."
    )
    confidence_reliable: bool = Field(
        description="False => confidence below floor or no checkpoint; treat as provisional."
    )
    is_healthy: bool = False


class SeverityReport(BaseModel):
    """The quantitative grounding signal. Deterministic OpenCV, no neural network."""

    severity_pct: float = Field(ge=0.0, le=100.0, description="Damaged / leaf area")
    necrotic_pct: float = Field(ge=0.0, le=100.0, description="Brown-dead tissue share")
    chlorotic_pct: float = Field(ge=0.0, le=100.0, description="Yellowing tissue share")
    healthy_pct: float = Field(ge=0.0, le=100.0)
    lesion_count: int = Field(ge=0)
    largest_lesion_pct: float = Field(ge=0.0, le=100.0)
    leaf_coverage_pct: float = Field(
        ge=0.0, le=100.0, description="Leaf pixels / frame — low value means a poor photo."
    )
    action_tier: ActionTier
    quality_warnings: list[str] = Field(default_factory=list)

    def to_prompt_block(self) -> str:
        return (
            f"- Damaged leaf area (severity index): {self.severity_pct:.1f} %\n"
            f"  - necrotic (dead/brown): {self.necrotic_pct:.1f} %\n"
            f"  - chlorotic (yellowing): {self.chlorotic_pct:.1f} %\n"
            f"  - healthy green: {self.healthy_pct:.1f} %\n"
            f"- Discrete lesions detected: {self.lesion_count} "
            f"(largest covers {self.largest_lesion_pct:.1f} % of the leaf)\n"
            f"- Measured action tier: {self.action_tier.value.upper()}"
        )


class PlanStep(BaseModel):
    order: int = Field(ge=1)
    action: str
    timing: str = Field(description="When to do it, e.g. 'within 48 hours', 'day 7'")
    rationale: str = ""
    citations: list[str] = Field(
        default_factory=list, description="Evidence ids [S1]... this step rests on"
    )


class TreatmentPlan(BaseModel):
    summary: str
    urgency: ActionTier
    immediate_actions: list[PlanStep] = Field(default_factory=list)
    followup_actions: list[PlanStep] = Field(default_factory=list)
    chemical_options: list[str] = Field(
        default_factory=list,
        description="Actives with label rates. Every entry must carry a citation id.",
    )
    cultural_practices: list[str] = Field(default_factory=list)
    monitoring: str = ""
    cautions: list[str] = Field(default_factory=list)
    generated_by: str = Field(description="groq | gemini | template")
    unsupported_claims_removed: int = 0


class Evidence(BaseModel):
    id: str = Field(description="Citation handle used in the plan, e.g. 'S1'")
    text: str
    score: float = 0.0
    source: str = ""
    doc_type: str = ""
    crop: str = ""
    disease: str = ""


class Readiness(BaseModel):
    """Honest per-component status. Nothing here ever blocks a response."""

    checkpoint_loaded: bool = False
    vector_store: str = "none"  # pgvector | local-file | empty
    documents_indexed: int = 0
    llm_provider: str = "template"
    degraded: list[str] = Field(default_factory=list)


class AdvisoryResponse(BaseModel):
    diagnosis: Diagnosis
    severity: SeverityReport
    environment: EnvironmentContext
    plan: TreatmentPlan
    evidence: list[Evidence] = Field(default_factory=list)
    readiness: Readiness
    elapsed_ms: dict[str, float] = Field(default_factory=dict)
    debug_overlay_png_b64: str | None = None
    disclaimer: str = (
        "Automated advisory generated from a single photograph. Verify actives, dosages "
        "and pre-harvest intervals against the product label and your local agricultural "
        "extension officer before applying anything."
    )
