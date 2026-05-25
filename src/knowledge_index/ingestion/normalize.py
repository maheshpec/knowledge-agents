"""Text normalization for ingestion (SPEC §7.1).

Unicode NFC normalization, whitespace collapse, and best-effort encoding
detection for raw bytes. Kept dependency-free: pure stdlib so it runs in any
environment and is trivially unit-testable.
"""

from __future__ import annotations

import re
import unicodedata

# Collapse runs of spaces/tabs but preserve paragraph breaks (blank lines).
_INLINE_WS = re.compile(r"[ \t\f\v]+")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_EXTRA_BLANKS = re.compile(r"\n{3,}")


def detect_encoding(blob: bytes) -> str:
    """Best-effort encoding detection for a byte string.

    Tries a UTF-8 BOM, then strict UTF-8, then falls back to cp1252 (a common
    superset of latin-1 for Western text). Returns the encoding name; decoding
    itself is done by :func:`decode_bytes`.
    """
    if blob.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        blob.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def decode_bytes(blob: bytes) -> str:
    """Decode bytes to ``str`` using the detected encoding, never raising."""
    enc = detect_encoding(blob)
    return blob.decode(enc, errors="replace")


def normalize_text(text: str) -> str:
    """Apply NFC + whitespace collapse, preserving paragraph structure.

    - Unicode NFC so visually identical strings hash identically (matters for
      the embedding cache and the dedup MinHash).
    - Collapse inline whitespace runs; strip trailing line whitespace.
    - Collapse 3+ blank lines to a single blank line.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_WS.sub(" ", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _EXTRA_BLANKS.sub("\n\n", text)
    return text.strip()


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalization for near-duplicate detection only.

    Lowercase, strip all punctuation/whitespace boundaries to a single space.
    Not used for stored text — only to compute MinHash shingles.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


__all__ = ["detect_encoding", "decode_bytes", "normalize_text", "normalize_for_dedup"]
