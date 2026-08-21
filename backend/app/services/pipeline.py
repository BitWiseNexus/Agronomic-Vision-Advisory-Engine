"""The orchestrator. One class, used identically by the API routes and the notebooks.

Load order matters for perceived latency: the classifier (ResNet-50 weights) and the
embedding model are both multi-second loads, so they are built once at construction and
reused. `AgronomicPipeline` is therefore a long-lived singleton, not a per-request object.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.llm.advisor import generate_plan
from app.llm.provider import LLMProvider, resolve_provider
from app.rag.index import KnowledgeIndex, load_index
from app.rag.retriever import retrieve_evidence
from app.schemas.diagnosis import AdvisoryResponse, Readiness, SeverityReport
from app.schemas.environment import EnvironmentContext
from app.vision.classifier import DiseaseClassifier
from app.vision.preprocess import decode_image, encode_png_b64
from app.vision.severity import SeverityConfig, SeverityMasks, analyse, render_overlay

log = get_logger(__name__)


@dataclass
class _Timer:
    marks: dict[str, float] = field(default_factory=dict)
    _start: float = field(default_factory=time.perf_counter)

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.marks[name] = round((now - self._start) * 1000, 1)
        self._start = now


class AgronomicPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        load_classifier: bool = True,
        load_knowledge: bool = True,
        severity_config: SeverityConfig | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.severity_config = severity_config or SeverityConfig()

        self.classifier: DiseaseClassifier | None = None
        if load_classifier:
            try:
                self.classifier = DiseaseClassifier(self.settings)
            except Exception as exc:  # torch missing / OOM — severity still works
                log.error("Classifier could not be constructed (%s)", exc)

        self.knowledge: KnowledgeIndex | None = None
        if load_knowledge:
            try:
                self.knowledge = load_index(self.settings)
            except Exception as exc:
                log.error("Knowledge index could not be opened (%s)", exc)

        self.provider: LLMProvider = resolve_provider(self.settings)
        log.info(
            "Pipeline ready | classifier=%s checkpoint=%s vectors=%s llm=%s",
            self.classifier is not None,
            self.classifier.status.checkpoint_loaded if self.classifier else False,
            self.knowledge.backend if self.knowledge else "none",
            self.provider.name,
        )

    # -- readiness -----------------------------------------------------------
    def readiness(self) -> Readiness:
        degraded: list[str] = []
        checkpoint = bool(self.classifier and self.classifier.status.checkpoint_loaded)
        if not self.classifier:
            degraded.append("classifier unavailable — severity only")
        elif not checkpoint:
            degraded.append("no trained checkpoint — disease labels are placeholders")

        backend = self.knowledge.backend if self.knowledge else "none"
        count = self.knowledge.document_count if self.knowledge else 0
        if backend in {"empty", "none"} or count == 0:
            degraded.append("knowledge base empty — advisory is ungrounded")
        elif backend == "local-file":
            degraded.append("using the on-disk vector store (Postgres/pgvector not connected)")
        if self.knowledge and self.knowledge.embedding.degraded:
            degraded.append("hash embeddings in use — retrieval quality is reduced")
        if self.provider.name == "template":
            degraded.append("no LLM key — deterministic template advisor in use")

        return Readiness(
            checkpoint_loaded=checkpoint,
            vector_store=backend,
            documents_indexed=count,
            llm_provider=self.provider.name,
            degraded=degraded,
        )

    # -- the pipeline --------------------------------------------------------
    def severity_only(self, bgr: np.ndarray) -> tuple[SeverityReport, SeverityMasks]:
        return analyse(bgr, self.severity_config)

    def run(
        self,
        image: bytes | np.ndarray,
        environment: EnvironmentContext | None = None,
        *,
        include_overlay: bool = False,
        top_k: int = 3,
    ) -> AdvisoryResponse:
        timer = _Timer()
        environment = environment or EnvironmentContext()
        bgr = decode_image(image) if isinstance(image, (bytes, bytearray)) else image
        timer.lap("decode")

        # 1. quantitative severity (deterministic)
        severity, masks = analyse(bgr, self.severity_config)
        timer.lap("severity")

        # 2. disease classification
        if self.classifier is not None:
            diagnosis = self.classifier.predict(bgr, top_k=top_k)
        else:
            diagnosis = self._placeholder_diagnosis()
        timer.lap("classify")

        # 3. retrieval grounded on both signals plus local context
        evidence: list = []
        if self.knowledge is not None:
            evidence, query = retrieve_evidence(
                self.knowledge, diagnosis, severity, environment, self.settings
            )
            log.debug("Retrieval query: %s", query)
        timer.lap("retrieve")

        # 4. synthesis
        plan = generate_plan(
            diagnosis, severity, environment, evidence, self.provider, self.settings
        )
        timer.lap("synthesize")

        overlay = None
        if include_overlay:
            overlay = encode_png_b64(render_overlay(masks))
            timer.lap("overlay")

        return AdvisoryResponse(
            diagnosis=diagnosis,
            severity=severity,
            environment=environment,
            plan=plan,
            evidence=evidence,
            readiness=self.readiness(),
            elapsed_ms=timer.marks,
            debug_overlay_png_b64=overlay,
        )

    @staticmethod
    def _placeholder_diagnosis():
        from app.schemas.diagnosis import ClassPrediction, Diagnosis

        return Diagnosis(
            top=ClassPrediction(
                label="Unknown___unavailable", crop="Unknown", disease="unavailable", confidence=0.0
            ),
            alternatives=[],
            checkpoint_loaded=False,
            confidence_reliable=False,
            is_healthy=False,
        )
