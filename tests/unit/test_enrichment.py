"""Tests for enrichment (SPEC §7.3)."""

from common.schemas import Chunk
from harness.cache.prompt_cache import count_breakpoints
from knowledge_index.enrichment import (
    ContextualEnricher,
    NullEnricher,
    TitleEnricher,
    build_enricher,
    embedding_text,
)
from knowledge_index.ingestion.base import ParsedDoc


def _doc() -> ParsedDoc:
    return ParsedDoc(
        doc_id="d1", text="Full document text about cats and dogs.", metadata={"title": "Animals"}
    )


def _chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="c1", doc_id="d1", text="Cats purr.", metadata={"header_path": ["Cats"]}),
    ]


async def test_null_enricher_is_noop():
    chunks = _chunks()
    out = await NullEnricher().enrich(_doc(), chunks)
    assert out[0].context is None


async def test_title_enricher_prepends_title_and_header():
    out = await TitleEnricher().enrich(_doc(), _chunks())
    assert out[0].context == "Animals › Cats"


def test_embedding_text_prepends_context():
    c = Chunk(chunk_id="c", doc_id="d", text="body", context="ctx")
    assert embedding_text(c) == "ctx\n\nbody"
    c2 = Chunk(chunk_id="c", doc_id="d", text="body")
    assert embedding_text(c2) == "body"


async def test_contextual_enricher_caches_document_block():
    captured: list[list[dict]] = []

    async def fake_complete(blocks):
        captured.append(blocks)
        return "This chunk is about cats."

    enricher = ContextualEnricher(completion_fn=fake_complete)
    out = await enricher.enrich(_doc(), _chunks())
    assert out[0].context == "This chunk is about cats."
    # the document block carries exactly one cache breakpoint; the chunk tail does not
    blocks = captured[0]
    assert "<document>" in blocks[0]["text"]
    assert count_breakpoints(blocks) == 1
    assert "cache_control" not in blocks[1]


def test_build_enricher_registry():
    assert isinstance(build_enricher("title"), TitleEnricher)
    assert isinstance(build_enricher("null"), NullEnricher)


async def test_summary_enricher_prepends_summary():
    from knowledge_index.enrichment import SummaryEnricher

    async def fake_complete(blocks):
        return "  A one-line summary.  "

    out = await SummaryEnricher(completion_fn=fake_complete).enrich(_doc(), _chunks())
    assert out[0].context == "A one-line summary."


def test_build_enricher_summary():
    from knowledge_index.enrichment import SummaryEnricher

    assert isinstance(build_enricher("summary"), SummaryEnricher)
