"""Typed runtime settings (SPEC §10.5).

All secrets and infra endpoints are read from the environment via this single
`Settings` model. No literal API keys live anywhere in the repo. Instantiate
once at process start (`get_settings()`) and pass via dependency injection —
never re-read inside hot paths.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration sourced from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM providers ---
    anthropic_api_key: str = ""
    openai_api_key: str | None = None  # fallback only

    # --- Embeddings & reranking ---
    voyage_api_key: str = ""
    cohere_api_key: str = ""

    # --- Vector store (Qdrant) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # --- Observability (LangSmith) ---
    langsmith_api_key: str | None = None
    langsmith_project: str = "knowledge-agent"

    # --- Graph store (Phase 3) ---
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None

    # --- Cache backends ---
    embedding_cache_path: str = ".cache/embeddings.sqlite"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    Cached so the .env file is read exactly once. Tests can clear the cache via
    `get_settings.cache_clear()`.
    """
    return Settings()


__all__ = ["Settings", "get_settings"]
