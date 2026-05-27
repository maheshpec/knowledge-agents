"""Tests for chunking strategies (SPEC §7.2)."""

import pytest

from knowledge_index.chunking import (
    CHUNKER_REGISTRY,
    LateChunkingChunker,
    MarkdownHeaderChunker,
    PropositionalChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceWindowChunker,
    build_chunker,
    split_sentences,
)
from knowledge_index.ingestion.base import ParsedDoc

# Fixed three-sentence document reused by the golden-output tests below.
_GOLDEN_TEXT = "Cats purr when content. Dogs bark at strangers. Birds sing at dawn."


def _doc(text: str) -> ParsedDoc:
    return ParsedDoc(doc_id="doc-1", text=text, metadata={"title": "T", "acl": ["alice"]})


def test_recursive_chunker_splits_and_sets_parent():
    text = " ".join(f"token{i}" for i in range(400))
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=20).chunk(_doc(text))
    assert len(chunks) > 1
    assert all(c.parent_id == "doc-1" for c in chunks)
    assert all(c.acl == ["alice"] for c in chunks)
    # deterministic, content-addressed ids
    again = RecursiveChunker(chunk_size=200, chunk_overlap=20).chunk(_doc(text))
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in again]


def test_recursive_chunker_exposes_config():
    ch = RecursiveChunker(chunk_size=500, chunk_overlap=75)
    assert ch.config == {"chunk_size": 500, "chunk_overlap": 75, "separators": None}
    assert ch.name == "recursive"


def test_markdown_header_chunker_records_header_path():
    md = "# Top\n\n" + "alpha " * 50 + "\n\n## Sub\n\n" + "beta " * 50
    chunks = MarkdownHeaderChunker(chunk_size=200, chunk_overlap=10).chunk(_doc(md))
    assert chunks
    assert any(c.metadata.get("header_path") for c in chunks)


def test_markdown_header_chunker_falls_back_without_headers():
    chunks = MarkdownHeaderChunker().chunk(_doc("plain text no headers here"))
    assert len(chunks) == 1


def test_build_chunker_registry():
    assert set(CHUNKER_REGISTRY) >= {
        "recursive",
        "markdown_header",
        "semantic",
        "sentence_window",
        "late_chunking",
        "propositional",
    }
    ch = build_chunker("recursive", chunk_size=300)
    assert ch.config["chunk_size"] == 300


# --- shared sentence splitter ------------------------------------------------


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("   ") == []


# --- semantic ----------------------------------------------------------------


def test_semantic_chunker_exposes_breakpoint_config():
    ch = SemanticChunker(threshold=0.8, breakpoint_type="standard_deviation")
    assert ch.config == {"threshold": 0.8, "breakpoint_type": "standard_deviation"}
    assert ch.name == "semantic"


def test_semantic_chunker_rejects_unknown_breakpoint_type():
    with pytest.raises(ValueError):
        SemanticChunker(breakpoint_type="bogus")


# --- sentence_window ---------------------------------------------------------


def test_sentence_window_one_chunk_per_sentence_with_window():
    chunks = SentenceWindowChunker(window_size=1).chunk(_doc(_GOLDEN_TEXT))
    assert [c.text for c in chunks] == [
        "Cats purr when content.",
        "Dogs bark at strangers.",
        "Birds sing at dawn.",
    ]
    # First chunk's window covers itself + the next sentence (no left neighbour).
    assert chunks[0].metadata["window"] == "Cats purr when content. Dogs bark at strangers."
    # Middle chunk's window spans both neighbours.
    assert chunks[1].metadata["window"] == _GOLDEN_TEXT
    assert all(c.metadata["window_size"] == 1 for c in chunks)


def test_sentence_window_zero_window_is_just_the_sentence():
    chunks = SentenceWindowChunker(window_size=0).chunk(_doc(_GOLDEN_TEXT))
    assert chunks[1].metadata["window"] == "Dogs bark at strangers."


def test_sentence_window_rejects_negative_window():
    with pytest.raises(ValueError):
        SentenceWindowChunker(window_size=-1)


# --- late_chunking -----------------------------------------------------------


def _stub_embed(texts: list[str]) -> list[list[float]]:
    # Deterministic 2-d vector per text: (length, word count).
    return [[float(len(t)), float(len(t.split()))] for t in texts]


def test_late_chunking_groups_sentences_and_pools_mean():
    ch = LateChunkingChunker(pool_method="mean", sentences_per_chunk=2, embed_fn=_stub_embed)
    chunks = ch.chunk(_doc(_GOLDEN_TEXT))
    # Three sentences, grouped 2 + 1.
    assert [c.text for c in chunks] == [
        "Cats purr when content. Dogs bark at strangers.",
        "Birds sing at dawn.",
    ]
    s = split_sentences(_GOLDEN_TEXT)
    v0, v1 = _stub_embed([s[0]])[0], _stub_embed([s[1]])[0]
    expected = [(v0[0] + v1[0]) / 2, (v0[1] + v1[1]) / 2]
    assert chunks[0].embedding == expected
    # Single-sentence group pools to that sentence's own vector.
    assert chunks[1].embedding == _stub_embed([s[2]])[0]


def test_late_chunking_max_pooling():
    ch = LateChunkingChunker(pool_method="max", sentences_per_chunk=3, embed_fn=_stub_embed)
    chunks = ch.chunk(_doc(_GOLDEN_TEXT))
    assert len(chunks) == 1
    s = split_sentences(_GOLDEN_TEXT)
    vecs = _stub_embed(s)
    assert chunks[0].embedding == [max(v[0] for v in vecs), max(v[1] for v in vecs)]


def test_late_chunking_without_embedder_leaves_embedding_unset():
    chunks = LateChunkingChunker(sentences_per_chunk=1).chunk(_doc(_GOLDEN_TEXT))
    assert len(chunks) == 3
    assert all(c.embedding is None for c in chunks)


def test_late_chunking_rejects_bad_params():
    with pytest.raises(ValueError):
        LateChunkingChunker(pool_method="median")
    with pytest.raises(ValueError):
        LateChunkingChunker(sentences_per_chunk=0)


# --- propositional -----------------------------------------------------------


def test_propositional_parses_json_array():
    captured: dict[str, str] = {}

    def fake_complete(prompt: str) -> str:
        captured["prompt"] = prompt
        return '["Cats purr.", "Dogs bark."]'

    chunks = PropositionalChunker(completion_fn=fake_complete).chunk(_doc(_GOLDEN_TEXT))
    assert [c.text for c in chunks] == ["Cats purr.", "Dogs bark."]
    # Full document is handed to the model.
    assert _GOLDEN_TEXT in captured["prompt"]
    assert all(c.parent_id == "doc-1" and c.acl == ["alice"] for c in chunks)


def test_propositional_parses_fenced_json_and_bulleted_fallback():
    fenced = PropositionalChunker(completion_fn=lambda p: '```json\n["A.", "B."]\n```').chunk(
        _doc(_GOLDEN_TEXT)
    )
    assert [c.text for c in fenced] == ["A.", "B."]

    bullets = PropositionalChunker(
        completion_fn=lambda p: "- First fact.\n- Second fact.\n2. Third fact."
    ).chunk(_doc(_GOLDEN_TEXT))
    assert [c.text for c in bullets] == ["First fact.", "Second fact.", "Third fact."]


def test_propositional_empty_doc_skips_llm():
    called = False

    def fake_complete(prompt: str) -> str:
        nonlocal called
        called = True
        return "[]"

    assert PropositionalChunker(completion_fn=fake_complete).chunk(_doc("   ")) == []
    assert called is False


def test_propositional_config_exposes_model():
    assert PropositionalChunker(model="claude-sonnet-4-6").config == {"model": "claude-sonnet-4-6"}


# --- golden output -----------------------------------------------------------
#
# Frozen full-output snapshots for the deterministic chunkers. If a change to
# splitting/grouping logic alters these, the diff is intentional — update the
# golden values in the same commit.


def test_golden_sentence_window():
    chunks = SentenceWindowChunker(window_size=1).chunk(_doc(_GOLDEN_TEXT))
    golden = [
        {
            "text": "Cats purr when content.",
            "window": "Cats purr when content. Dogs bark at strangers.",
        },
        {
            "text": "Dogs bark at strangers.",
            "window": _GOLDEN_TEXT,
        },
        {
            "text": "Birds sing at dawn.",
            "window": "Dogs bark at strangers. Birds sing at dawn.",
        },
    ]
    assert [{"text": c.text, "window": c.metadata["window"]} for c in chunks] == golden


def test_golden_late_chunking():
    ch = LateChunkingChunker(pool_method="mean", sentences_per_chunk=2, embed_fn=_stub_embed)
    chunks = ch.chunk(_doc(_GOLDEN_TEXT))
    golden = [
        {
            "text": "Cats purr when content. Dogs bark at strangers.",
            "embedding": [23.0, 4.0],
        },
        {"text": "Birds sing at dawn.", "embedding": [19.0, 4.0]},
    ]
    assert [{"text": c.text, "embedding": c.embedding} for c in chunks] == golden


def test_golden_propositional():
    # Realistic decontextualised propositions — rewrites, not source substrings.
    chunks = PropositionalChunker(
        completion_fn=lambda p: '["A cat purrs when it is content.", "A dog barks at strangers."]'
    ).chunk(_doc(_GOLDEN_TEXT))
    assert [c.text for c in chunks] == [
        "A cat purrs when it is content.",
        "A dog barks at strangers.",
    ]
    # Propositions are rewrites, not substrings, so they carry no span.
    assert all(c.metadata["span"] is None for c in chunks)
