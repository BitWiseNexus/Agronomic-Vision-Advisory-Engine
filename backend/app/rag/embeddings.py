"""Embeddings — local and free.

`BAAI/bge-small-en-v1.5` runs on CPU, needs no API key and has no rate limit, which is
the whole reason it was chosen over a hosted embedding endpoint. Downloading it needs
internet *once*; after that it is cached. If it cannot be loaded at all (offline first
run, no HF cache) we fall back to a deterministic hashing embedder so ingestion and
retrieval remain exercisable — retrieval quality drops to keyword-ish, and the caller is
told via `EmbeddingInfo.degraded`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class EmbeddingInfo:
    model_name: str
    dim: int
    degraded: bool
    note: str = ""


def _hash_embed(text: str, dim: int) -> list[float]:
    """Deterministic bag-of-hashed-tokens vector, L2-normalised.

    Not semantic — it is a signed random projection of token counts — but it is stable,
    dependency-free and good enough to prove the plumbing end to end.
    """
    vector = np.zeros(dim, dtype=np.float32)
    tokens = _TOKEN_RE.findall(text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector.tolist()


def _build_hash_embedding(dim: int) -> Any:
    from llama_index.core.embeddings import BaseEmbedding

    class HashEmbedding(BaseEmbedding):
        """LlamaIndex-compatible wrapper around `_hash_embed`."""

        embed_dim: int = dim

        @classmethod
        def class_name(cls) -> str:
            return "hash_embedding"

        def _get_query_embedding(self, query: str) -> list[float]:
            return _hash_embed(query, self.embed_dim)

        def _get_text_embedding(self, text: str) -> list[float]:
            return _hash_embed(text, self.embed_dim)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

        async def _aget_text_embedding(self, text: str) -> list[float]:
            return self._get_text_embedding(text)

    return HashEmbedding(embed_dim=dim)


_model_cache: tuple[Any, EmbeddingInfo] | None = None


def get_embedding_model(settings: Settings | None = None) -> tuple[Any, EmbeddingInfo]:
    """Cached — loading a sentence-transformer takes seconds, so never do it per request.

    Hand-rolled rather than `@lru_cache` because `Settings` is a pydantic model and so is
    unhashable; there is only ever one embedding model per process anyway.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    settings = settings or get_settings()
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        model = HuggingFaceEmbedding(model_name=settings.embedding_model, device="cpu")
        probe = model.get_text_embedding("leaf")
        info = EmbeddingInfo(settings.embedding_model, len(probe), degraded=False)
        if info.dim != settings.embedding_dim:
            log.warning(
                "EMBEDDING_DIM=%d but %s produces %d dims — using the real value; "
                "update .env and the pgvector table to match.",
                settings.embedding_dim,
                settings.embedding_model,
                info.dim,
            )
        log.info("Embeddings: %s (%d dims, local)", info.model_name, info.dim)
        _model_cache = (model, info)
    except Exception as exc:
        log.warning("Local embedding model unavailable (%s) — falling back to hash embeddings", exc)
        dim = settings.embedding_dim
        _model_cache = (
            _build_hash_embedding(dim),
            EmbeddingInfo("hash-embedding", dim, degraded=True, note=str(exc)),
        )
    return _model_cache
