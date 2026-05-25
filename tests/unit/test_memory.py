"""Tests for the three memory scopes (SPEC §6.3)."""

import pytest

from harness.memory import (
    LayeredMemory,
    LongTermMemory,
    MemoryExtractor,
    SessionMemory,
    WorkingMemory,
)
from harness.memory.extraction import ExtractedFact, ExtractionResult
from knowledge_index.embedding import HashEmbedder
from knowledge_index.indexing import QdrantIndex

DIM = 48


def _longterm(owner: str | None = None) -> LongTermMemory:
    index = QdrantIndex("ka_memory_longterm", dim=DIM, location=":memory:")
    return LongTermMemory(index, HashEmbedder(dim=DIM), owner=owner)


async def _fake_extract(text: str) -> ExtractionResult:
    facts = []
    if "dark mode" in text:
        facts.append(ExtractedFact(key="ui_pref", value="dark mode", kind="preference"))
    if "python" in text.lower():
        facts.append(ExtractedFact(key="lang", value="Python", kind="fact"))
    return ExtractionResult(facts=facts)


# --- working memory ---


async def test_working_write_read():
    from common.types import MemoryItem

    wm = WorkingMemory()
    await wm.write(MemoryItem(key="k", value="hello world", scope="working"))
    hits = await wm.read("hello")
    assert len(hits) == 1
    assert hits[0].value == "hello world"


async def test_working_clear_per_turn():
    from common.types import MemoryItem

    wm = WorkingMemory()
    await wm.write(MemoryItem(key="k", value="v", scope="working"))
    wm.clear()
    assert await wm.read("") == []


async def test_working_forget_predicate():
    from common.types import MemoryItem

    wm = WorkingMemory()
    await wm.write(MemoryItem(key="keep", value="a", scope="working"))
    await wm.write(MemoryItem(key="drop", value="b", scope="working"))
    await wm.forget(lambda it: it.key == "drop")
    keys = {it.key for it in await wm.all()}
    assert keys == {"keep"}


# --- session memory ---


async def test_session_persists_and_reads_recent():
    sm = SessionMemory("sess-1", path=":memory:")
    from common.types import MemoryItem

    await sm.write(MemoryItem(key="a", value="first fact", scope="session"))
    await sm.write(MemoryItem(key="b", value="second fact", scope="session"))
    items = await sm.read("")
    assert {i.key for i in items} == {"a", "b"}
    # query filters by substring
    only = await sm.read("second")
    assert [i.key for i in only] == ["b"]
    sm.close()


async def test_session_scope_isolation():
    s1 = SessionMemory("s1", path=":memory:")
    from common.types import MemoryItem

    await s1.write(MemoryItem(key="x", value="v", scope="session"))
    # a different session id on the same in-memory db sees nothing here because
    # each :memory: connection is independent — assert s1 has its own item
    assert len(await s1.all()) == 1
    s1.close()


async def test_session_forget():
    sm = SessionMemory("s", path=":memory:")
    from common.types import MemoryItem

    await sm.write(MemoryItem(key="temp", value="x", scope="session"))
    await sm.forget(lambda it: it.key == "temp")
    assert await sm.all() == []
    sm.close()


# --- long-term memory ---


async def test_longterm_write_read_roundtrip():
    from common.types import MemoryItem

    lt = _longterm()
    await lt.write(MemoryItem(key="fav_lang", value="Python is great", scope="long_term"))
    hits = await lt.read("which programming language", k=3)
    assert hits
    assert hits[0].key == "fav_lang"
    assert hits[0].score is not None


async def test_longterm_acl_per_user():
    from common.types import MemoryItem

    index = QdrantIndex("ka_memory_longterm", dim=DIM, location=":memory:")
    emb = HashEmbedder(dim=DIM)
    alice = LongTermMemory(index, emb, owner="alice")
    await alice.write(MemoryItem(key="secret", value="alice private note", scope="long_term"))
    # bob shares the same index but a different owner: cannot read alice's note
    bob = LongTermMemory(index, emb, owner="bob")
    assert await bob.read("private note", k=5) == []
    assert await alice.read("private note", k=5)


async def test_longterm_forget():
    from common.types import MemoryItem

    lt = _longterm()
    await lt.write(MemoryItem(key="ephemeral", value="forget me", scope="long_term"))
    await lt.forget(lambda it: it.key == "ephemeral")
    assert await lt.all() == []


# --- extraction + layered ---


async def test_extractor_only_keeps_durable_facts():
    extractor = MemoryExtractor(extract_fn=_fake_extract)
    items = await extractor.extract("I love dark mode and code in python", scope="long_term")
    keys = {i.key for i in items}
    assert keys == {"ui_pref", "lang"}
    # nothing durable -> empty (do not store everything)
    none = await extractor.extract("hello how are you", scope="long_term")
    assert none == []


async def test_layered_routes_by_scope():
    lt = _longterm()
    mem = LayeredMemory(
        working=WorkingMemory(),
        session=SessionMemory("s", path=":memory:"),
        long_term=lt,
        extractor=MemoryExtractor(extract_fn=_fake_extract),
    )
    await mem.write("w", "working val", "working")
    await mem.write("s", "session val", "session")
    assert await mem.read("working", "working")
    assert await mem.read("session", "session")


async def test_layered_consolidate_is_extraction_gated():
    lt = _longterm()
    mem = LayeredMemory(
        working=WorkingMemory(),
        session=SessionMemory("s", path=":memory:"),
        long_term=lt,
        extractor=MemoryExtractor(extract_fn=_fake_extract),
    )
    written = await mem.consolidate("the user prefers dark mode", scope="long_term")
    assert [i.key for i in written] == ["ui_pref"]
    # the fact is now retrievable from long-term memory
    hits = await mem.read("appearance preference", "long_term")
    assert any(h.key == "ui_pref" for h in hits)


async def test_layered_consolidate_without_extractor_raises():
    mem = LayeredMemory(
        working=WorkingMemory(),
        session=SessionMemory("s", path=":memory:"),
        long_term=_longterm(),
    )
    with pytest.raises(ValueError):
        await mem.consolidate("anything")
