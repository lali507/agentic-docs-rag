"""Central configuration, read once from the environment / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def enable_netfree_ssl_if_available() -> None:
    """Support networks that re-sign TLS traffic (the starter project needs this on NetFree).

    The package is optional: without it, nothing changes.
    """
    try:
        from netfree_unstrict_ssl import unstrict_ssl
    except ImportError:
        return
    unstrict_ssl()


def _resolve_paths(raw: str) -> tuple[Path, ...]:
    paths = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            path = Path(part).expanduser()
            paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
    return tuple(paths)


@dataclass(frozen=True)
class Settings:
    cohere_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index: str = "agentic-docs"
    pinecone_namespace: str = "agentic-docs"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # The documents and questions are mostly Hebrew, so the multilingual model is required.
    embed_model: str = "embed-multilingual-v3.0"
    embed_dimension: int = 1024
    llm_model: str = "command-a-03-2025"

    docs_roots: tuple[Path, ...] = (PROJECT_ROOT / "data" / "sample_project",)
    knowledge_path: Path = PROJECT_ROOT / "data" / "structured" / "knowledge.json"
    extraction_cache_path: Path = PROJECT_ROOT / "data" / "structured" / "extraction_cache.json"

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 6
    # Chunks scoring below this are dropped.
    min_score: float = 0.20
    # A top score below this triggers another search from a different angle.
    confidence_threshold: float = 0.35
    max_search_attempts: int = 2
    max_question_chars: int = 2000
    # The free Cohere key allows 20 chat calls a minute, hence about 3.5 seconds between calls.
    extraction_delay_seconds: float = 3.5


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    defaults = Settings()
    return Settings(
        cohere_api_key=os.getenv("COHERE_API_KEY", ""),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_index=os.getenv("PINECONE_INDEX", defaults.pinecone_index),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", defaults.pinecone_namespace),
        docs_roots=_resolve_paths(os.getenv("DOCS_ROOTS", "")) or defaults.docs_roots,
        llm_model=os.getenv("COHERE_LLM_MODEL", defaults.llm_model),
        min_score=float(os.getenv("RAG_MIN_SCORE", defaults.min_score)),
        confidence_threshold=float(
            os.getenv("RAG_CONFIDENCE_THRESHOLD", defaults.confidence_threshold)
        ),
        top_k=int(os.getenv("RAG_TOP_K", defaults.top_k)),
    )
