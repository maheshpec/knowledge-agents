"""Reciprocal Rank Fusion (SPEC §7.6.4).

RRF combines rankings by *position*, not score, which makes it robust to the
incomparable score scales of dense (cosine) vs. sparse (BM25) retrievers. Each
candidate contributes ``1 / (k + rank)`` from every list it appears in; the
constant ``k`` (default 60, per Cormack et al.) damps the influence of top ranks
so a single retriever cannot dominate.
"""

from __future__ import annotations

from common.schemas import RetrievalCandidate
from harness.observability.tracing import traced
from knowledge_index.retrieval.fusion.base import candidate_key

DEFAULT_RRF_K = 60


class RRFFuser:
    """Reciprocal Rank Fusion across retriever result lists."""

    name = "rrf"

    def __init__(self, k: int = DEFAULT_RRF_K) -> None:
        if k <= 0:
            raise ValueError("RRF k must be positive")
        self._k = k

    @traced(span_name="retrieval.fusion.rrf")
    async def fuse(self, results: list[list[RetrievalCandidate]]) -> list[RetrievalCandidate]:
        scores: dict[str, float] = {}
        best: dict[str, RetrievalCandidate] = {}

        for ranked in results:
            for position, candidate in enumerate(ranked):
                # Trust the retriever's own ``rank`` when set (1-based); otherwise
                # fall back to list position so callers needn't pre-rank.
                rank = candidate.rank if candidate.rank > 0 else position + 1
                key = candidate_key(candidate)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self._k + rank)
                # Keep the highest-scoring original instance as the carrier of
                # chunk text/embedding/metadata.
                if key not in best or candidate.score > best[key].score:
                    best[key] = candidate

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        fused: list[RetrievalCandidate] = []
        for new_rank, (key, fused_score) in enumerate(ordered, start=1):
            carrier = best[key]
            fused.append(
                carrier.model_copy(
                    update={"score": fused_score, "retriever": self.name, "rank": new_rank}
                )
            )
        return fused


__all__ = ["DEFAULT_RRF_K", "RRFFuser"]
