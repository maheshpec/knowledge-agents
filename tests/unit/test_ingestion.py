"""Tests for ingestion: normalize, dedup, markdown parsing (SPEC §7.1)."""

from common.types import MimeType
from knowledge_index.ingestion import (
    MarkdownParser,
    MinHashDeduplicator,
    jaccard,
    mime_from_path,
    normalize_text,
    parse_blob,
)
from knowledge_index.ingestion.normalize import decode_bytes, detect_encoding

# --- normalization ---


def test_normalize_collapses_whitespace_and_blanks():
    text = "a   b\t\tc\r\n\n\n\nd  \n"
    out = normalize_text(text)
    assert out == "a b c\n\nd"


def test_normalize_nfc_makes_equivalent_strings_equal():
    composed = "café"  # é as single codepoint
    decomposed = "café"  # e + combining acute
    assert composed != decomposed
    assert normalize_text(composed) == normalize_text(decomposed)


def test_detect_and_decode_encoding():
    assert detect_encoding("héllo".encode()) == "utf-8"
    assert detect_encoding(b"\xef\xbb\xbfhi") == "utf-8-sig"
    # invalid utf-8 falls back, never raises
    assert decode_bytes(b"\xff\xfe abc")


# --- markdown parser ---


async def test_markdown_parser_extracts_structure():
    md = b"# Title\n\nIntro para.\n\n## Section\n\n| a | b |\n| - | - |\n| 1 | 2 |\n\n```py\nx=1\n```\n"
    doc = await parse_blob(md, MimeType.MARKDOWN)
    kinds = [e.kind for e in doc.structure]
    assert "heading" in kinds
    assert "table" in kinds
    assert "code" in kinds
    assert doc.metadata["title"] == "Title"
    # table preserved as markdown (pipes intact)
    table = next(e for e in doc.structure if e.kind == "table")
    assert "|" in table.text


async def test_markdown_doc_id_is_content_addressed():
    a = await MarkdownParser().parse(b"same bytes")
    b = await MarkdownParser().parse(b"same bytes")
    c = await MarkdownParser().parse(b"other bytes")
    assert a.doc_id == b.doc_id
    assert a.doc_id != c.doc_id


def test_mime_from_path():
    assert mime_from_path("a.pdf") == MimeType.PDF
    assert mime_from_path("a.md") == MimeType.MARKDOWN
    assert mime_from_path("a.unknownext") == MimeType.UNKNOWN


# --- dedup ---


def test_minhash_flags_near_duplicates():
    base = " ".join(f"word{i}" for i in range(200))
    near = base + " word200 word201"  # >90% overlap
    far = " ".join(f"different{i}" for i in range(200))
    dedup = MinHashDeduplicator(threshold=0.8)
    dedup.add("a", base)
    matches = dedup.add("b", near)
    dedup.add("c", far)
    assert "a" in matches
    clusters = dedup.clusters()
    assert any({"a", "b"} <= set(members) for members in clusters.values())
    # the far doc is not clustered with a/b
    assert not any("c" in members for members in clusters.values())


def test_jaccard_identical_and_disjoint():
    assert jaccard("the quick brown fox jumps", "the quick brown fox jumps") == 1.0
    assert jaccard("aaa bbb ccc ddd eee", "fff ggg hhh iii jjj") == 0.0
