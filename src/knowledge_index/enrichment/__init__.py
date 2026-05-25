"""Enrichment package (SPEC §7.3): contextual retrieval + cheaper variants."""

from __future__ import annotations

from typing import Any

from knowledge_index.enrichment.base import Enricher, embedding_text
from knowledge_index.enrichment.enrichers import (
    ContextualEnricher,
    NullEnricher,
    SummaryEnricher,
    TitleEnricher,
)

ENRICHER_REGISTRY: dict[str, type] = {
    "null": NullEnricher,
    "contextual": ContextualEnricher,
    "title": TitleEnricher,
    "summary": SummaryEnricher,
}


def build_enricher(name: str, **params: Any) -> Enricher:
    """Instantiate an enricher by registry name (mirrors components.yaml)."""
    if name not in ENRICHER_REGISTRY:
        raise KeyError(f"unknown enricher '{name}'; known: {sorted(ENRICHER_REGISTRY)}")
    return ENRICHER_REGISTRY[name](**params)  # type: ignore[return-value]


__all__ = [
    "Enricher",
    "embedding_text",
    "NullEnricher",
    "ContextualEnricher",
    "TitleEnricher",
    "SummaryEnricher",
    "ENRICHER_REGISTRY",
    "build_enricher",
]
