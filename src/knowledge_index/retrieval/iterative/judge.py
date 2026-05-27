"""Hop judge for the iterative retriever (SPEC §7.6.7).

After each retrieval round the loop asks a *judge* one question: is the answer
already in the accumulated evidence, and if not, what follow-up query would close
the gap? The judge is the only LLM call per hop, so it is injected as a
``CompleteFn`` (``str -> str``) — exactly like the router/query-ops — letting the
whole retriever run offline under test.

:class:`HopDecision` is the parsed verdict. :class:`LLMHopJudge` is the default
implementation; tests inject a scripted judge implementing the :class:`HopJudge`
protocol to make multi-hop behaviour deterministic.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from common.schemas import Query, RetrievalCandidate
from knowledge_index.retrieval.query_ops.base import CompleteFn, default_completer

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"

# Evidence snippets are truncated before they reach the prompt: the judge needs
# enough to spot a gap, not the full chunk text. Bounds the token cost per hop.
_MAX_EVIDENCE_CHUNKS = 8
_SNIPPET_CHARS = 240

JUDGE_PROMPT = """You are the controller of a multi-hop retrieval loop. Given the \
ORIGINAL question and the evidence gathered so far, decide whether the evidence is \
sufficient to answer the question fully.

Return ONLY a JSON object (no prose, no code fences):
  {{"done": true|false,
    "next_query": "<a focused follow-up search query, or empty string if done>",
    "reasoning": "<one short sentence>"}}

Rules:
- If the evidence already answers the question, set done=true and next_query="".
- If a fact is still missing, set done=false and write next_query as a search \
query that targets *only the missing fact* — not a paraphrase of the original.
- next_query must be a fresh query, not one already tried.

ORIGINAL question: {query}

Evidence so far:
{evidence}"""


class HopDecision(BaseModel):
    """The judge's verdict for one hop of the iterative loop."""

    done: bool
    next_query: str = ""
    reasoning: str = ""

    @property
    def follow_up(self) -> str:
        """The trimmed follow-up query, or empty when there is none to run."""
        return self.next_query.strip()


@runtime_checkable
class HopJudge(Protocol):
    """Decides whether to stop and what to ask next (SPEC §7.6.7)."""

    async def judge(self, original: Query, evidence: list[RetrievalCandidate]) -> HopDecision: ...


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _parse_decision(raw: str) -> HopDecision:
    """Parse the judge reply into a :class:`HopDecision`, tolerating fences."""
    payload = _strip_code_fence(raw)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"hop judge did not return valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("hop judge must return a JSON object")
    return HopDecision.model_validate(data)


def _format_evidence(evidence: list[RetrievalCandidate]) -> str:
    """Render the highest-ranked evidence as a compact numbered list for the prompt."""
    if not evidence:
        return "(none yet)"
    lines = []
    for i, cand in enumerate(evidence[:_MAX_EVIDENCE_CHUNKS], start=1):
        snippet = cand.chunk.text.strip().replace("\n", " ")[:_SNIPPET_CHARS]
        lines.append(f"[{i}] {snippet}")
    return "\n".join(lines)


class LLMHopJudge:
    """LLM-backed hop judge: one classification call per hop (SPEC §7.6.7)."""

    name = "llm_hop_judge"

    def __init__(self, complete: CompleteFn | None = None) -> None:
        self._complete = complete or default_completer(DEFAULT_JUDGE_MODEL)

    async def judge(self, original: Query, evidence: list[RetrievalCandidate]) -> HopDecision:
        prompt = JUDGE_PROMPT.format(query=original.raw, evidence=_format_evidence(evidence))
        return _parse_decision(await self._complete(prompt))


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "JUDGE_PROMPT",
    "HopDecision",
    "HopJudge",
    "LLMHopJudge",
]
