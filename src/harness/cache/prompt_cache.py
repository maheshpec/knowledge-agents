"""Anthropic prompt-cache helpers (SPEC §6.12 tier 1).

The actual caching is performed server-side by the Anthropic API when a content
block carries a ``cache_control`` marker. These helpers build correctly-shaped
blocks so the system prompt, skills, and long retrieved contexts are cached
across calls (critical for affordable contextual retrieval — SPEC §7.3).
"""

from __future__ import annotations

from typing import Any

# Anthropic currently supports a small number of cache breakpoints per request.
MAX_CACHE_BREAKPOINTS = 4


def cacheable_text_block(text: str, *, cache: bool = True) -> dict[str, Any]:
    """Build a single text content block, optionally marked for prompt caching."""
    block: dict[str, Any] = {"type": "text", "text": text}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def build_cached_system(segments: list[str], *, cache_last: bool = True) -> list[dict[str, Any]]:
    """Build a system prompt as content blocks with a cache breakpoint at the end.

    Anthropic caches everything up to and including the block carrying
    ``cache_control``. Place stable content (frozen system prompt, skills) first
    and set the breakpoint on the final stable block so the whole prefix is
    reused on subsequent calls.
    """
    if not segments:
        return []
    blocks: list[dict[str, Any]] = [cacheable_text_block(s, cache=False) for s in segments]
    if cache_last:
        blocks[-1] = cacheable_text_block(segments[-1], cache=True)
    return blocks


def count_breakpoints(blocks: list[dict[str, Any]]) -> int:
    """Count how many content blocks carry a cache_control marker."""
    return sum(1 for b in blocks if "cache_control" in b)


__all__ = [
    "MAX_CACHE_BREAKPOINTS",
    "cacheable_text_block",
    "build_cached_system",
    "count_breakpoints",
]
