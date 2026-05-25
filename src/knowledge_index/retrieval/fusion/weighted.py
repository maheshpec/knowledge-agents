"""Score-normalized weighted fusion (SPEC §7.6.4).

The alternative to RRF: min-max normalize each retriever's scores into ``[0, 1]``
(so BM25 and cosine become comparable), then take a weighted sum per chunk.
Weights are configured per retriever *by name*; unlisted retrievers default to
weight 1.0.
"""

from __future__ import annotations

from common.schemas import RetrievalCandidate
from harness.observability.tracing import traced
from knowledge_index.retrieval.fusion.base import candidate_key


def _normalize(candidates: list[RetrievalCandidate]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1]; a flat list maps everything to 1.0."""
    if not candidates:
        return {}
    scores = [c.score for c in candidates]
    lo, hi = min(scores), max(scores)
    spread = hi - lo
    out: dict[str, float] = {}
    for c in candidates:
        out[candidate_key(c)] = 1.0 if spread == 0 else (c.score - lo) / spread
    return out


class WeightedFuser:
    """Weighted sum of min-max-normalized per-retriever scores."""

    name = "weighted"

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or {}

    def _weight_for(self, candidates: list[RetrievalCandidate]) -> float:
        retriever = candidates[0].retriever if candidates else ""
        return self._weights.get(retriever, 1.0)

    @traced(span_name="retrieval.fusion.weighted")
    async def fuse(self, results: list[list[RetrievalCandidate]]) -> list[RetrievalCandidate]:
        scores: dict[str, float] = {}
        best: dict[str, RetrievalCandidate] = {}

        for ranked in results:
            weight = self._weight_for(ranked)
            normalized = _normalize(ranked)
            for candidate in ranked:
                key = candidate_key(candidate)
                scores[key] = scores.get(key, 0.0) + weight * normalized[key]
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


__all__ = ["WeightedFuser"]
