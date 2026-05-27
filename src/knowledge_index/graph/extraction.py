"""Entity + relation extraction for graph construction (SPEC §7.7).

Two extractors share one interface:

* :class:`HeuristicExtractor` — deterministic, network-free. Entities are
  capitalized noun phrases / multi-word proper nouns; triples come from a small
  set of verb patterns ("X acquired Y", "X founded by Y"). It is the dev/test
  default and the offline fallback when no LLM is configured.
* :class:`LLMExtractor` — LLM-based NER + constrained-schema triplet extraction,
  using the same injectable ``completion_fn`` pattern as the contextual enricher
  (SPEC §7.3) so tests can run it without a network.

Both return canonical types from :mod:`knowledge_index.graph.base`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from knowledge_index.graph.base import Entity, Triple, normalize_entity

CompletionFn = Callable[[list[dict[str, Any]]], Awaitable[str]]

# A capitalized token run, optionally chained ("Acme Corp", "Jane Doe"). Lower-case
# connectors (of/and/&) may sit between runs so "Bank of England" stays one entity.
_PROPER_NOUN = re.compile(
    r"\b[A-Z][\w'-]*(?:\s+(?:of|and|&|the|de|von|van)\s+[A-Z][\w'-]*|\s+[A-Z][\w'-]*)*"
)

# Relation cues: (left entity) <verb phrase> (right entity). Predicate is normalized.
_RELATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(.+?)\s+(?:was\s+)?acquired\s+(?:by\s+)?(.+)", re.I), "acquired"),
    (re.compile(r"(.+?)\s+(?:was\s+)?founded\s+by\s+(.+)", re.I), "founded_by"),
    (re.compile(r"(.+?)\s+founded\s+(.+)", re.I), "founded"),
    (re.compile(r"(.+?)\s+is\s+(?:the\s+)?(?:ceo|cto|head|president)\s+of\s+(.+)", re.I), "leads"),
    (re.compile(r"(.+?)\s+(?:is\s+)?(?:a\s+)?subsidiary\s+of\s+(.+)", re.I), "subsidiary_of"),
    (re.compile(r"(.+?)\s+(?:is\s+)?(?:a\s+)?(?:part|division)\s+of\s+(.+)", re.I), "part_of"),
    (re.compile(r"(.+?)\s+(?:is\s+)?located\s+in\s+(.+)", re.I), "located_in"),
    (re.compile(r"(.+?)\s+(?:is\s+)?based\s+in\s+(.+)", re.I), "located_in"),
    (re.compile(r"(.+?)\s+(?:is\s+)?owned\s+by\s+(.+)", re.I), "owned_by"),
    (re.compile(r"(.+?)\s+partnered\s+with\s+(.+)", re.I), "partnered_with"),
]

_STOPWORDS = {"The", "This", "That", "These", "Those", "It", "He", "She", "They", "A", "An"}


def _first_entity(text: str) -> str | None:
    """Pick the leading proper-noun phrase from a relation operand."""
    m = _PROPER_NOUN.search(text.strip())
    if not m:
        return None
    name = m.group(0).strip(" .,:;")
    return name or None


class HeuristicExtractor:
    """Deterministic regex extractor — no LLM, no network (dev/test default)."""

    name = "heuristic"

    async def extract_entities(self, text: str) -> list[Entity]:
        seen: dict[str, Entity] = {}
        for m in _PROPER_NOUN.finditer(text):
            name = m.group(0).strip(" .,:;")
            if not name or name in _STOPWORDS or len(name) < 2:
                continue
            key = normalize_entity(name)
            seen.setdefault(key, Entity(name=name, type="concept", key=key))
        return list(seen.values())

    async def extract_triples(self, text: str) -> list[Triple]:
        triples: list[Triple] = []
        # Split on sentence/clause boundaries so a pattern matches within one clause.
        for clause in re.split(r"[.;\n]", text):
            clause = clause.strip()
            if not clause:
                continue
            for pat, predicate in _RELATION_PATTERNS:
                m = pat.match(clause)
                if not m:
                    continue
                subj = _first_entity(m.group(1))
                obj = _first_entity(m.group(2))
                if subj and obj and normalize_entity(subj) != normalize_entity(obj):
                    triples.append(Triple(subject=subj, predicate=predicate, object=obj))
                break  # first matching pattern wins for the clause
        return triples


# --- LLM-based extraction (SPEC §7.7: NER + constrained triplet extraction) ---

_ENTITY_PROMPT = (
    "Extract the named entities and key domain terms from the text below. "
    "Return ONLY a JSON array of objects with keys 'name' and 'type' "
    "(type one of: person, org, place, product, concept). No prose.\n\n"
    "<text>\n{text}\n</text>"
)

_TRIPLE_PROMPT = (
    "Extract subject-predicate-object relation triples from the text below. "
    "Use short snake_case predicates (e.g. acquired, founded_by, located_in). "
    "Return ONLY a JSON array of objects with keys 'subject', 'predicate', "
    "'object'. No prose.\n\n<text>\n{text}\n</text>"
)


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    """Best-effort parse of an LLM JSON-array reply (tolerates code fences/prose)."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?|\n?```$", "", raw).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


class LLMExtractor:
    """LLM NER + triplet extraction; reuses the enricher ``completion_fn`` pattern.

    Falls back to :class:`HeuristicExtractor` for any text where the model returns
    nothing usable, so the graph is never empty just because a call degraded.
    """

    name = "llm"

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5-20251001",
        completion_fn: CompletionFn | None = None,
        api_key: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self._completion_fn = completion_fn
        self._api_key = api_key
        self.max_tokens = max_tokens
        self._client: Any = None
        self._fallback = HeuristicExtractor()

    def _default_completion_fn(self) -> CompletionFn:
        async def _complete(blocks: list[dict[str, Any]]) -> str:
            if self._client is None:
                from anthropic import AsyncAnthropic  # type: ignore

                self._client = AsyncAnthropic(api_key=self._api_key)
            resp = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": blocks}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

        return _complete

    async def _complete(self, prompt: str) -> str:
        fn = self._completion_fn or self._default_completion_fn()
        return await fn([{"type": "text", "text": prompt}])

    async def extract_entities(self, text: str) -> list[Entity]:
        raw = await self._complete(_ENTITY_PROMPT.format(text=text))
        entities: dict[str, Entity] = {}
        for d in _parse_json_array(raw):
            name = str(d.get("name", "")).strip()
            if not name:
                continue
            key = normalize_entity(name)
            entities.setdefault(
                key, Entity(name=name, type=str(d.get("type", "concept")) or "concept", key=key)
            )
        if entities:
            return list(entities.values())
        return await self._fallback.extract_entities(text)

    async def extract_triples(self, text: str) -> list[Triple]:
        raw = await self._complete(_TRIPLE_PROMPT.format(text=text))
        triples: list[Triple] = []
        for d in _parse_json_array(raw):
            s, p, o = (
                str(d.get("subject", "")).strip(),
                str(d.get("predicate", "")).strip(),
                str(d.get("object", "")).strip(),
            )
            if s and p and o and normalize_entity(s) != normalize_entity(o):
                triples.append(Triple(subject=s, predicate=p, object=o))
        if triples:
            return triples
        return await self._fallback.extract_triples(text)


__all__ = ["CompletionFn", "HeuristicExtractor", "LLMExtractor"]
