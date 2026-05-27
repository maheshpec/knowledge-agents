"""Lost-in-the-middle reordering post-processor (SPEC §7.6.6, §6.6).

Long-context models attend most strongly to the *start* and *end* of their input
and least to the middle. This post-processor takes the rerank-ordered candidates
(best→worst) and interleaves them so the strongest evidence sits at *both* ends of
the block and the weakest is buried in the middle::

    ranked:  [1, 2, 3, 4, 5]   ->   reordered: [1, 3, 5, 4, 2]

The same policy is applied at context-packing time (``harness.context.packer``);
exposing it as a registry post-processor lets the evolutionary loop (§8) select it
independently of the packer and measure its effect in isolation.
"""

from __future__ import annotations

from common.schemas import Query, RetrievalCandidate
from harness.observability.tracing import traced


class LostInTheMiddleReorder:
    """Reorder candidates so the most relevant sit at both ends (SPEC §6.6)."""

    name = "lost_in_the_middle"

    @traced(span_name="retrieval.post.lost_in_the_middle")
    async def process(
        self, query: Query, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        if len(candidates) <= 2:
            return candidates

        # Candidates are assumed ranked best→worst. Best to the top, second-best to
        # the bottom, and so on inward — weakest evidence ends up in the middle.
        front: list[RetrievalCandidate] = []
        back: list[RetrievalCandidate] = []
        for i, cand in enumerate(candidates):
            (front if i % 2 == 0 else back).append(cand)
        ordered = front + back[::-1]

        return [c.model_copy(update={"rank": i}) for i, c in enumerate(ordered, start=1)]


__all__ = ["LostInTheMiddleReorder"]
