"""Default context-packing policy (SPEC §6.6).

Order: frozen+cached system prompt → skills → memory hits → retrieved chunks
(reordered so the most-relevant sit at *both* ends of the block, combating
lost-in-the-middle) → scratchpad → recent conversation turns last. The retrieved
block is trimmed worst-first to fit ``budget_tokens``.

Each chunk is rendered with its ``chunk_id`` label so the citation enforcer
(SPEC §6.13) can reference exactly the evidence the model saw.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage

from common.schemas import RetrievalCandidate, RetrievalResult
from common.types import MemoryItem
from harness.cache.prompt_cache import cacheable_text_block
from harness.context.base import Skill, estimate_tokens


def reorder_for_lost_in_middle(
    candidates: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    """Place the highest-scored candidates at both ends of the list.

    Input is assumed ranked best→worst. Best goes to the top, second-best to the
    bottom, and so on inward — so the weakest evidence ends up in the middle,
    where models attend least (SPEC §6.6).
    """
    front: list[RetrievalCandidate] = []
    back: list[RetrievalCandidate] = []
    for i, cand in enumerate(candidates):
        (front if i % 2 == 0 else back).append(cand)
    return front + back[::-1]


def render_candidate(cand: RetrievalCandidate) -> str:
    """Render one candidate as a labelled, citable evidence block."""
    chunk = cand.chunk
    header = f"[{chunk.chunk_id}]"
    body = f"{chunk.context}\n{chunk.text}" if chunk.context else chunk.text
    return f"{header}\n{body}"


class DefaultPacker:
    """The Phase 1 context packer (SPEC §6.6)."""

    name = "default"

    def __init__(
        self, *, evidence_header: str = "Retrieved context (cite claims by [id]):"
    ) -> None:
        self._evidence_header = evidence_header

    def render_preamble(self, skills: list[Skill], memory_hits: list[MemoryItem]) -> str:
        """Render selected skills + memory hits as a guidance preamble (SPEC §6.6).

        Used by the orchestrator's answer step to put Phase 2 context (the skills
        the registry selected for this query's intent, and any long-term memory
        hits) in front of the question, in the same order ``pack`` would place
        them. Returns an empty string when neither is present.
        """
        blocks: list[str] = []
        if skills:
            blocks.append(
                "\n\n".join(f"## Skill: {s.name}\n{s.instructions}" for s in skills)
            )
        if memory_hits:
            mem = "\n".join(f"- {item.key}: {item.value}" for item in memory_hits)
            blocks.append(f"Relevant memory:\n{mem}")
        return "\n\n".join(blocks)

    def fit_candidates(
        self, candidates: list[RetrievalCandidate], budget_tokens: int
    ) -> list[RetrievalCandidate]:
        """Drop worst-ranked candidates until the evidence block fits the budget."""
        kept = list(candidates)
        while kept:
            rendered = "\n\n".join(render_candidate(c) for c in kept)
            if estimate_tokens(rendered) <= budget_tokens:
                break
            kept.pop()  # candidates are ranked best→worst; drop the tail
        return kept

    def order_evidence(
        self, candidates: list[RetrievalCandidate], budget_tokens: int
    ) -> list[RetrievalCandidate]:
        """Budget-trim then end-load candidates — the final evidence ordering.

        Exposed so the orchestrator feeds the citation enforcer exactly the
        passages (and order) the packer would place in the prompt.
        """
        return reorder_for_lost_in_middle(self.fit_candidates(candidates, budget_tokens))

    def pack(
        self,
        system: str,
        skills: list[Skill],
        memory_hits: list[MemoryItem],
        retrieval: RetrievalResult | None,
        scratchpad: str,
        messages: list[BaseMessage],
        budget_tokens: int,
    ) -> list[BaseMessage]:
        out: list[BaseMessage] = []

        # 1. Frozen system prompt + skills, marked for Anthropic prompt caching.
        system_text = system
        if skills:
            skill_text = "\n\n".join(f"## Skill: {s.name}\n{s.instructions}" for s in skills)
            system_text = f"{system_text}\n\n{skill_text}"
        out.append(SystemMessage(content=[cacheable_text_block(system_text, cache=True)]))

        # 2. Memory hits (Phase 1: usually empty).
        if memory_hits:
            mem_text = "\n".join(f"- {item.key}: {item.value}" for item in memory_hits)
            out.append(SystemMessage(content=f"Relevant memory:\n{mem_text}"))

        # 3. Retrieved evidence: budget-trim worst-first, then end-load the best.
        if retrieval is not None and retrieval.candidates:
            ordered = self.order_evidence(retrieval.candidates, budget_tokens)
            evidence = "\n\n".join(render_candidate(c) for c in ordered)
            out.append(SystemMessage(content=f"{self._evidence_header}\n\n{evidence}"))

        # 4. Scratchpad (intermediate reasoning / observations).
        if scratchpad.strip():
            out.append(SystemMessage(content=f"Scratchpad:\n{scratchpad.strip()}"))

        # 5. Recent conversation turns last (most recent = closest to generation).
        out.extend(messages)
        return out


__all__ = ["DefaultPacker", "reorder_for_lost_in_middle", "render_candidate"]
