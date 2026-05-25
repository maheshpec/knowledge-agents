"""Maximal Marginal Relevance diversifier (SPEC §7.6.6).

MMR greedily builds a result set that balances relevance to the query against
novelty relative to already-selected results::

    MMR = λ · sim(query, doc) − (1 − λ) · max_{s∈selected} sim(doc, s)

λ=1 is pure relevance (no diversification); λ=0 is pure diversity. The default
0.5 splits the difference.

Document vectors come from ``chunk.embedding`` when the dense retriever populated
them; otherwise the injected ``embed`` callable fills the gap. The query vector is
always produced via ``embed`` since :class:`Query` carries no embedding.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from common.schemas import Chunk, Query, RetrievalCandidate
from harness.observability.tracing import traced
from knowledge_index.retrieval.post.base import cosine

EmbedFn = Callable[[str], Awaitable[list[float]]]

DEFAULT_MMR_LAMBDA = 0.5


class MMRDiversifier:
    """Re-order candidates by Maximal Marginal Relevance."""

    name = "mmr"

    def __init__(
        self,
        embed: EmbedFn,
        *,
        lambda_: float = DEFAULT_MMR_LAMBDA,
        top_k: int | None = None,
    ) -> None:
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError("MMR lambda must be in [0, 1]")
        self._embed = embed
        self._lambda = lambda_
        self._top_k = top_k

    async def _doc_vec(self, chunk: Chunk) -> list[float]:
        if chunk.embedding:
            return chunk.embedding
        return await self._embed(
            f"{chunk.context}\n\n{chunk.text}" if chunk.context else chunk.text
        )

    @traced(span_name="retrieval.post.mmr")
    async def process(
        self, query: Query, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        if len(candidates) <= 1:
            return candidates

        limit = self._top_k or len(candidates)
        query_vec = await self._embed(query.raw)
        doc_vecs = [await self._doc_vec(c.chunk) for c in candidates]
        relevance = [cosine(query_vec, dv) for dv in doc_vecs]

        remaining = list(range(len(candidates)))
        selected: list[int] = []

        while remaining and len(selected) < limit:
            best_idx = remaining[0]
            best_score = float("-inf")
            for i in remaining:
                if selected:
                    redundancy = max(cosine(doc_vecs[i], doc_vecs[s]) for s in selected)
                else:
                    redundancy = 0.0
                mmr = self._lambda * relevance[i] - (1.0 - self._lambda) * redundancy
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [
            candidates[idx].model_copy(update={"rank": new_rank})
            for new_rank, idx in enumerate(selected, start=1)
        ]


__all__ = ["DEFAULT_MMR_LAMBDA", "EmbedFn", "MMRDiversifier"]
