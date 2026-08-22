"""Central configuration. Everything is optional so the app boots with no .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---------------------------------------------------------------
    llm_provider: str = "auto"  # groq | gemini | template | auto
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1800
    llm_timeout_seconds: float = 60.0

    # --- Vector store ------------------------------------------------------
    database_url: str = ""
    pgvector_table: str = "agro_knowledge"

    # --- Embeddings --------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- Vision ------------------------------------------------------------
    checkpoint_path: str = "models/checkpoints/resnet50_plantvillage.pt"
    device: str = "auto"
    classifier_confidence_floor: float = 0.45

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = 8
    retrieval_keep: int = 4

    # --- App ---------------------------------------------------------------
    log_level: str = "INFO"

    # --- Derived paths -----------------------------------------------------
    @property
    def root(self) -> Path:
        return PROJECT_ROOT

    @property
    def knowledge_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "knowledge"

    @property
    def samples_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "samples"

    @property
    def storage_dir(self) -> Path:
        return PROJECT_ROOT / "storage"

    @property
    def checkpoint_file(self) -> Path:
        path = Path(self.checkpoint_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
