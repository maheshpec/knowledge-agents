"""Span extraction post-processor (SPEC §7.6.6).

Finds the most query-relevant contiguous sub-span within each chunk and trims the
chunk down to it. Small focused spans reduce the tokens the generator must read
and combat "needle in a haystack" dilution when a chunk is mostly irrelevant.

The extractor is deterministic and offline: it splits the chunk into sentences,
scores a sliding window of up to ``max_sentences`` by query-term overlap, and
keeps the best-scoring window. The original chunk text and the char offsets of the
extracted span are preserved on ``chunk.metadata`` (``span_original_text`` /
``span_offsets``) so nothing is lost. A chunk with no query-term overlap is left
untouched (better to keep the whole chunk than to guess).
"""

from __future__ import annotations

import re

from common.schemas import Query, RetrievalCandidate
from harness.observability.tracing import traced

# Sentence boundary: terminator followed by whitespace. Keeps offsets recoverable.
_SENTENCE = re.compile(r"[^.!?]+[.!?]*\s*")
_WORD = re.compile(r"\w+")

DEFAULT_MAX_SENTENCES = 3


def _query_terms(query: Query) -> set[str]:
    """Lower-cased word set from the raw query plus any rewrites."""
    text = " ".join([query.raw, *query.rewrites])
    return {w.lower() for w in _WORD.findall(text)}


class SpanExtractor:
    """Trim each chunk to its most query-relevant sentence window (SPEC §7.6.6)."""

    name = "span_extractor"

    def __init__(self, max_sentences: int = DEFAULT_MAX_SENTENCES) -> None:
        if max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")
        self._max_sentences = max_sentences

    def _best_span(self, text: str, terms: set[str]) -> tuple[int, int] | None:
        """Return (start, end) char offsets of the best window, or None to keep all."""
        # (offset, sentence_text) for each sentence, preserving exact char spans.
        sents: list[tuple[int, str]] = [(m.start(), m.group()) for m in _SENTENCE.finditer(text)]
        if len(sents) <= 1:
            return None  # nothing to trim

        def overlap(sentence: str) -> int:
            words = {w.lower() for w in _WORD.findall(sentence)}
            return len(words & terms)

        scores = [overlap(s) for _, s in sents]
        if not any(scores):
            return None  # no signal; keep the whole chunk

        best_score = -1
        best: tuple[int, int] | None = None
        window = min(self._max_sentences, len(sents))
        for start in range(len(sents) - window + 1):
            score = sum(scores[start : start + window])
            if score > best_score:
                best_score = score
                span_start = sents[start][0]
                last_off, last_txt = sents[start + window - 1]
                best = (span_start, last_off + len(last_txt))
        return best

    @traced(span_name="retrieval.post.span_extractor")
    async def process(
        self, query: Query, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        terms = _query_terms(query)
        if not terms:
            return candidates

        out: list[RetrievalCandidate] = []
        for candidate in candidates:
            chunk = candidate.chunk
            span = self._best_span(chunk.text, terms)
            if span is None:
                out.append(candidate)
                continue
            start, end = span
            extracted = chunk.text[start:end].strip()
            if not extracted or extracted == chunk.text.strip():
                out.append(candidate)
                continue
            new_meta = {
                **chunk.metadata,
                "span_original_text": chunk.text,
                "span_offsets": (start, end),
            }
            new_chunk = chunk.model_copy(update={"text": extracted, "metadata": new_meta})
            out.append(candidate.model_copy(update={"chunk": new_chunk}))
        return out


__all__ = ["DEFAULT_MAX_SENTENCES", "SpanExtractor"]
