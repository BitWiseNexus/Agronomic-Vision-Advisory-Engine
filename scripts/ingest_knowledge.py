r"""Build the vector index from data/knowledge/.

    .\.venv\Scripts\python.exe scripts\ingest_knowledge.py
    .\.venv\Scripts\python.exe scripts\ingest_knowledge.py --reset --local
"""

from __future__ import annotations

import argparse

from _path import bootstrap

bootstrap()

from app.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.rag.ingest import ingest_knowledge  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the agricultural knowledge corpus")
    parser.add_argument("--reset", action="store_true", help="Truncate the pgvector table first")
    parser.add_argument(
        "--local", action="store_true", help="Force the on-disk store even if Postgres is up"
    )
    args = parser.parse_args()

    configure_logging("INFO")
    settings = get_settings()
    print(f"Corpus:     {settings.knowledge_dir}")
    print(f"Embeddings: {settings.embedding_model}")

    _, report = ingest_knowledge(settings, prefer_pgvector=not args.local, reset=args.reset)
    print(f"\n{report}")


if __name__ == "__main__":
    main()
