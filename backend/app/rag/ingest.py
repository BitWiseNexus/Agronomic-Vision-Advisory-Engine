"""Corpus ingestion: data/knowledge/*.md -> chunked, embedded nodes.

Front matter becomes node metadata, which is what makes filtered retrieval ("only
tomato documents") possible later. Chunk size 768 with 96 overlap keeps a full
treatment recommendation inside one chunk in most extension bulletins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.rag.embeddings import get_embedding_model
from app.rag.index import KnowledgeIndex, ensure_no_default_llm, try_pgvector_store

log = get_logger(__name__)

CHUNK_SIZE = 768
CHUNK_OVERLAP = 96
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class IngestReport:
    documents: int
    nodes: int
    backend: str
    embedding_model: str
    skipped: list[str]

    def __str__(self) -> str:
        return (
            f"{self.documents} documents -> {self.nodes} nodes into '{self.backend}' "
            f"using {self.embedding_model}"
            + (f" (skipped: {', '.join(self.skipped)})" if self.skipped else "")
        )


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Minimal YAML front-matter reader — flat `key: value` pairs and `[a, b]` lists.

    Deliberately hand-rolled: pulling in PyYAML for ten lines of metadata is not worth
    the dependency, and the format here is fixed by our own authoring convention.
    """
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        cleaned = value.strip().strip("\"'")
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = ", ".join(part.strip().strip("\"'") for part in cleaned[1:-1].split(","))
        metadata[key.strip().lower()] = cleaned
    return metadata, raw[match.end() :]


def _read_pdf(path: Path) -> str:
    try:
        from llama_index.readers.file import PDFReader

        return "\n\n".join(doc.text for doc in PDFReader().load_data(file=path))
    except Exception as exc:
        log.warning("Could not read %s (%s)", path.name, exc)
        return ""


def load_documents(knowledge_dir: Path) -> tuple[list[Any], list[str]]:
    from llama_index.core import Document

    documents: list[Any] = []
    skipped: list[str] = []

    if not knowledge_dir.exists():
        return documents, [f"{knowledge_dir} does not exist"]

    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        if path.suffix.lower() == ".pdf":
            text, metadata = _read_pdf(path), {}
        else:
            metadata, text = _parse_front_matter(path.read_text(encoding="utf-8", errors="replace"))

        if not text.strip():
            skipped.append(path.name)
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    "source": metadata.get("source", path.stem.replace("_", " ")),
                    "file_name": path.name,
                    "doc_type": metadata.get("doc_type", "extension_bulletin"),
                    "crop": metadata.get("crop", "general"),
                    "disease": metadata.get("disease", "general"),
                    "region": metadata.get("region", "general"),
                    "title": metadata.get("title", path.stem.replace("_", " ").title()),
                },
                # Keep metadata out of the embedded text except the parts that carry
                # real retrieval signal — file paths and ids only add noise.
                excluded_embed_metadata_keys=["file_name", "source", "region", "doc_type"],
            )
        )
    return documents, skipped


def ingest_knowledge(
    settings: Settings | None = None,
    knowledge_dir: Path | None = None,
    prefer_pgvector: bool = True,
    reset: bool = False,
) -> tuple[KnowledgeIndex, IngestReport]:
    """(Re)build the index from the corpus on disk."""
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter

    settings = settings or get_settings()
    ensure_no_default_llm()
    directory = knowledge_dir or settings.knowledge_dir

    documents, skipped = load_documents(directory)
    if not documents:
        raise FileNotFoundError(
            f"No ingestible documents in {directory}. Add .md/.txt/.pdf extension "
            "material, or run scripts/make_samples.py for the seed corpus."
        )

    embed_model, embed_info = get_embedding_model(settings)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)

    store = try_pgvector_store(settings, embed_info.dim) if prefer_pgvector else None
    if store is not None:
        if reset:
            _truncate_pg_table(settings)
        context = StorageContext.from_defaults(vector_store=store)
        index = VectorStoreIndex(nodes, storage_context=context, embed_model=embed_model)
        backend = "pgvector"
    else:
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(settings.storage_dir))
        backend = "local-file"

    report = IngestReport(len(documents), len(nodes), backend, embed_info.model_name, skipped)
    log.info("Ingest complete: %s", report)
    return (
        KnowledgeIndex(index, backend, embed_info, document_count=len(nodes)),
        report,
    )


def _truncate_pg_table(settings: Settings) -> None:
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.database_url)
        with engine.begin() as connection:
            connection.execute(
                text(f'TRUNCATE TABLE "data_{settings.pgvector_table}"')  # noqa: S608
            )
        log.info("Truncated data_%s before re-ingest", settings.pgvector_table)
    except Exception as exc:
        log.warning("Could not truncate the pgvector table (%s)", exc)
