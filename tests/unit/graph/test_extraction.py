"""Tests for entity/relation extraction (SPEC §7.7)."""

from __future__ import annotations

from knowledge_index.graph.base import normalize_entity
from knowledge_index.graph.extraction import HeuristicExtractor, LLMExtractor


async def test_heuristic_extracts_proper_noun_entities():
    ex = HeuristicExtractor()
    ents = await ex.extract_entities("Acme Corp acquired Globex in Berlin.")
    keys = {e.key for e in ents}
    assert "acme corp" in keys
    assert "globex" in keys
    assert "berlin" in keys


async def test_heuristic_skips_sentence_initial_stopwords():
    ex = HeuristicExtractor()
    ents = await ex.extract_entities("The report was filed.")
    # "The" is a stopword; "report"/"was"/"filed" are lower-case -> no entities.
    assert ents == []


async def test_heuristic_extracts_acquired_triple():
    ex = HeuristicExtractor()
    triples = await ex.extract_triples("Acme Corp acquired Globex last year.")
    assert any(
        t.predicate == "acquired"
        and normalize_entity(t.subject) == "acme corp"
        and normalize_entity(t.object) == "globex"
        for t in triples
    )


async def test_heuristic_extracts_founded_by_triple():
    ex = HeuristicExtractor()
    triples = await ex.extract_triples("Globex was founded by Jane Doe.")
    assert any(
        t.predicate == "founded_by" and normalize_entity(t.object) == "jane doe" for t in triples
    )


async def test_llm_extractor_parses_json_and_falls_back():
    async def fake_entities(blocks):
        return '```json\n[{"name": "OpenAI", "type": "org"}]\n```'

    ex = LLMExtractor(completion_fn=fake_entities)
    ents = await ex.extract_entities("anything")
    assert [e.name for e in ents] == ["OpenAI"]

    # Empty/garbage reply -> heuristic fallback kicks in.
    async def empty(blocks):
        return "no entities here"

    ex2 = LLMExtractor(completion_fn=empty)
    ents2 = await ex2.extract_entities("Acme Corp shipped a product.")
    assert any(e.key == "acme corp" for e in ents2)


async def test_llm_extractor_triples_from_json():
    async def fake(blocks):
        return '[{"subject": "Acme", "predicate": "acquired", "object": "Globex"}]'

    ex = LLMExtractor(completion_fn=fake)
    triples = await ex.extract_triples("x")
    assert triples[0].predicate == "acquired"
