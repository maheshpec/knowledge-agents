"""Cohere cross-encoder reranker (SPEC §7.6.5).

Wraps Cohere Rerank 3. The async client is injected so the reranker is testable
offline; when omitted, a default ``cohere.AsyncClientV2`` is built lazily from
``Settings.cohere_api_key`` (the heavy import is deferred to first use).

The reranked relevance scores replace the candidates' fusion scores, and ranks
are renumbered 1..N over the surviving top-k.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from common.errors import RetrievalError
from common.schemas import RetrievalCandidate
from harness.observability.tracing import traced

# Cohere Rerank 3 family; override via the constructor as the model line evolves.
DEFAULT_COHERE_RERANK_MODEL = "rerank-v3.5"


@runtime_checkable
class SupportsCohereRerank(Protocol):
    """The slice of ``cohere.AsyncClientV2`` this reranker calls."""

    async def rerank(self, *, model: str, query: str, documents: list[str], top_n: int) -> Any: ...


def _doc_text(candidate: RetrievalCandidate) -> str:
    """Text handed to the reranker: contextual prefix (SPEC §7.3) + chunk text."""
    chunk = candidate.chunk
    if chunk.context:
        return f"{chunk.context}\n\n{chunk.text}"
    return chunk.text


class CohereReranker:
    """Rerank candidates with Cohere Rerank 3."""

    name = "cohere"

    def __init__(
        self,
        client: SupportsCohereRerank | None = None,
        *,
        model: str = DEFAULT_COHERE_RERANK_MODEL,
    ) -> None:
        self._client = client
        self._model = model

    def _get_client(self) -> SupportsCohereRerank:
        if self._client is None:
            import cohere  # lazy: avoid importing the SDK at module load

            from common.settings import get_settings

            self._client = cohere.AsyncClientV2(api_key=get_settings().cohere_api_key)
        return self._client

    @traced(span_name="retrieval.rerank.cohere")
    async def rerank(
        self, query: str, candidates: list[RetrievalCandidate], top_k: int
    ) -> list[RetrievalCandidate]:
        if not candidates:
            return []

        documents = [_doc_text(c) for c in candidates]
        response = await self._get_client().rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=min(top_k, len(documents)),
        )

        results = getattr(response, "results", None)
        if results is None:
            raise RetrievalError("Cohere rerank response missing 'results'")

        reranked: list[RetrievalCandidate] = []
        for new_rank, item in enumerate(results, start=1):
            original = candidates[item.index]
            reranked.append(
                original.model_copy(
                    update={
                        "score": float(item.relevance_score),
                        "retriever": self.name,
                        "rank": new_rank,
                    }
                )
            )
        return reranked


__all__ = ["DEFAULT_COHERE_RERANK_MODEL", "SupportsCohereRerank", "CohereReranker"]
