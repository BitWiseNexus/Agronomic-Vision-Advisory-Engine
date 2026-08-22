"""Vector index construction: pgvector when Postgres answers, on-disk store otherwise.

Both paths hand back the same `VectorStoreIndex`, so `retriever.py` never learns which
one it got. `KnowledgeIndex.backend` records the truth for the readiness matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.rag.embeddings import EmbeddingInfo, get_embedding_model

log = get_logger(__name__)


_llm_silenced = False


def ensure_no_default_llm() -> None:
    """LlamaIndex defaults to OpenAI for anything LLM-shaped. We do our own LLM calls
    over httpx, so pin a MockLLM and guarantee no surprise paid API traffic.

    Note the module-level flag rather than an `isinstance(LlamaSettings.llm, ...)` check:
    *reading* that property is what triggers the default resolution, which raises
    ImportError for llama-index-llms-openai. The guard must never read it.
    """
    global _llm_silenced
    if _llm_silenced:
        return
    from llama_index.core import Settings as LlamaSettings
    from llama_index.core.llms import MockLLM

    LlamaSettings.llm = MockLLM()
    _llm_silenced = True


@dataclass
class KnowledgeIndex:
    index: Any
    backend: str  # "pgvector" | "local-file" | "empty"
    embedding: EmbeddingInfo
    document_count: int = 0
    note: str = ""

    @property
    def is_empty(self) -> bool:
        return self.document_count == 0


def _pg_kwargs(database_url: str, settings: Settings, embed_dim: int) -> dict[str, Any]:
    parsed = urlparse(database_url.replace("+psycopg2", "").replace("+asyncpg", ""))
    return {
        "database": (parsed.path or "/agro").lstrip("/"),
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username or "agro",
        "password": parsed.password or "agro",
        "table_name": settings.pgvector_table,
        "embed_dim": embed_dim,
        "hnsw_kwargs": {
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    }


def try_pgvector_store(settings: Settings, embed_dim: int) -> Any | None:
    if not settings.database_url:
        log.info("DATABASE_URL not set — using the local file vector store.")
        return None
    try:
        from llama_index.vector_stores.postgres import PGVectorStore

        store = PGVectorStore.from_params(**_pg_kwargs(settings.database_url, settings, embed_dim))
        store._initialize()  # opens the connection and creates the table/extension
        log.info("Connected to pgvector table '%s'", settings.pgvector_table)
        return store
    except Exception as exc:
        log.warning("pgvector unavailable (%s) — falling back to the local file store", exc)
        return None


def load_index(settings: Settings | None = None, prefer_pgvector: bool = True) -> KnowledgeIndex:
    """Open the existing index. Never ingests — call `rag.ingest.ingest_knowledge` for that."""
    from llama_index.core import (
        StorageContext,
        VectorStoreIndex,
        load_index_from_storage,
    )

    settings = settings or get_settings()
    ensure_no_default_llm()
    embed_model, embed_info = get_embedding_model(settings)

    store = try_pgvector_store(settings, embed_info.dim) if prefer_pgvector else None
    if store is not None:
        index = VectorStoreIndex.from_vector_store(store, embed_model=embed_model)
        return KnowledgeIndex(index, "pgvector", embed_info, document_count=_pg_count(settings))

    persist_dir = settings.storage_dir
    if (persist_dir / "docstore.json").exists():
        try:
            context = StorageContext.from_defaults(persist_dir=str(persist_dir))
            index = load_index_from_storage(context, embed_model=embed_model)
            count = len(index.docstore.docs)
            log.info("Loaded local vector store from %s (%d nodes)", persist_dir, count)
            return KnowledgeIndex(index, "local-file", embed_info, document_count=count)
        except Exception as exc:
            log.warning("Local store at %s is unreadable (%s)", persist_dir, exc)

    log.warning("No index found. Run scripts/ingest_knowledge.py to populate it.")
    empty = VectorStoreIndex.from_documents([], embed_model=embed_model)
    return KnowledgeIndex(empty, "empty", embed_info, 0, "no documents indexed")


def _pg_count(settings: Settings) -> int:
    """Row count straight from SQL — cheaper and more honest than asking LlamaIndex."""
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.database_url)
        table = f"data_{settings.pgvector_table}"
        with engine.connect() as connection:
            result = connection.execute(text(f'SELECT count(*) FROM "{table}"'))  # noqa: S608
            return int(result.scalar() or 0)
    except Exception:
        return 0
