"""Concrete chunking strategies (SPEC §7.2).

``RecursiveChunker`` and ``MarkdownHeaderChunker`` are the Phase 1 defaults and
use ``langchain_text_splitters`` (a core dependency). The Phase 3 "chunker zoo"
adds embedding- and LLM-driven strategies — ``SemanticChunker``,
``SentenceWindowChunker``, ``LateChunkingChunker`` and ``PropositionalChunker``.
Heavy or networked dependencies (``langchain_experimental``, the Anthropic SDK,
an embedder) are imported/invoked lazily and are injectable, so this module
imports cleanly and every chunker is unit-testable without them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from common.schemas import Chunk
from knowledge_index.chunking.base import build_chunks, split_sentences
from knowledge_index.ingestion.base import ParsedDoc

# A sync embedding function (texts -> vectors). Matches the langchain
# ``Embeddings.embed_documents`` shape so a langchain embedder works directly.
EmbedFn = Callable[[list[str]], list[list[float]]]

# A sync completion function for ``PropositionalChunker`` (prompt -> model text).
CompletionFn = Callable[[str], str]

# Default markdown header levels to split on, coarse → fine.
_DEFAULT_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


class RecursiveChunker:
    """Character-recursive splitter (SPEC §7.2 default: 500/75 on markdown)."""

    name = "recursive"

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.config: dict[str, Any] = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separators": separators,
        }

    def _splitter(self):  # type: ignore[no-untyped-def]
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        kwargs: dict[str, Any] = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        if self.separators is not None:
            kwargs["separators"] = self.separators
        return RecursiveCharacterTextSplitter(**kwargs)

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        texts = self._splitter().split_text(doc.text)
        return build_chunks(doc, texts)


class MarkdownHeaderChunker:
    """Split by markdown headers, then recursively size each section (SPEC §7.2).

    Header path is recorded in each chunk's metadata so downstream packers and
    the ``TitleEnricher`` can reconstruct section context.
    """

    name = "markdown_header"

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
    ) -> None:
        self.headers_to_split_on = headers_to_split_on or _DEFAULT_HEADERS
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.config: dict[str, Any] = {
            "headers_to_split_on": [h[0] for h in self.headers_to_split_on],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )
        sections = header_splitter.split_text(doc.text)
        size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        texts: list[str] = []
        extra: list[dict[str, Any]] = []
        for section in sections:
            header_path = [v for v in section.metadata.values()]
            for piece in size_splitter.split_text(section.page_content):
                texts.append(piece)
                extra.append({"header_path": header_path})
        if not texts:
            # No headers present: fall back to plain recursive sizing.
            texts = size_splitter.split_text(doc.text)
            extra = [{} for _ in texts]
        return build_chunks(doc, texts, extra_metadata=extra)


# langchain SemanticChunker accepts these breakpoint statistics; the threshold
# amount is interpreted relative to whichever type is chosen.
_BREAKPOINT_TYPES = ("percentile", "standard_deviation", "interquartile", "gradient")


class SemanticChunker:
    """Embedding-based semantic splitter (lazy ``langchain_experimental``).

    Splits where the embedding distance between adjacent sentences exceeds a
    breakpoint computed by ``breakpoint_type`` at ``threshold`` (SPEC §7.2:
    "breakpoint type, threshold"). For ``percentile`` the threshold is a
    fraction in ``[0, 1]`` scaled to a percentile; for the other statistics it
    is passed through as the amount.
    """

    name = "semantic"

    def __init__(
        self,
        threshold: float = 0.75,
        breakpoint_type: str = "percentile",
        embedder: Any = None,
    ) -> None:
        if breakpoint_type not in _BREAKPOINT_TYPES:
            raise ValueError(
                f"unknown breakpoint_type '{breakpoint_type}'; known: {_BREAKPOINT_TYPES}"
            )
        self.threshold = threshold
        self.breakpoint_type = breakpoint_type
        self._embedder = embedder
        self.config: dict[str, Any] = {
            "threshold": threshold,
            "breakpoint_type": breakpoint_type,
        }

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        try:
            from langchain_experimental.text_splitter import SemanticChunker as _LCSemantic
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "SemanticChunker requires 'langchain_experimental'; install ingest extras"
            ) from e
        if self._embedder is None:
            raise ValueError("SemanticChunker requires an embeddings backend")
        # `percentile` expects a 0-100 amount; other statistics take the raw value.
        amount = self.threshold * 100 if self.breakpoint_type == "percentile" else self.threshold
        splitter = _LCSemantic(
            self._embedder,
            breakpoint_threshold_type=self.breakpoint_type,
            breakpoint_threshold_amount=amount,
        )
        texts = splitter.split_text(doc.text)
        return build_chunks(doc, texts)


class SentenceWindowChunker:
    """One-sentence chunks plus a surrounding context window (SPEC §7.2).

    Each retrievable chunk is a single sentence (precise embedding target); the
    ``window`` metadata holds that sentence joined with ``window_size`` neighbour
    sentences on either side, so a parent-style expander can hand the generator
    the wider context at synthesis time.
    """

    name = "sentence_window"

    def __init__(self, window_size: int = 1) -> None:
        if window_size < 0:
            raise ValueError("window_size must be >= 0")
        self.window_size = window_size
        self.config: dict[str, Any] = {"window_size": window_size}

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        sentences = split_sentences(doc.text)
        extra: list[dict[str, Any]] = []
        for i in range(len(sentences)):
            lo = max(0, i - self.window_size)
            hi = min(len(sentences), i + self.window_size + 1)
            extra.append({"window": " ".join(sentences[lo:hi]), "window_size": self.window_size})
        return build_chunks(doc, sentences, extra_metadata=extra)


def _pool(vectors: list[list[float]], method: str) -> list[float]:
    """Pool a group of equal-length vectors element-wise (mean or max)."""
    if not vectors:
        return []
    dim = len(vectors[0])
    if method == "max":
        return [max(v[j] for v in vectors) for j in range(dim)]
    if method == "mean":
        n = len(vectors)
        return [sum(v[j] for v in vectors) / n for j in range(dim)]
    raise ValueError(f"unknown pool_method '{method}'; expected 'mean' or 'max'")


class LateChunkingChunker:
    """Late chunking: embed at sentence granularity, pool by chunk (SPEC §7.2).

    True late chunking embeds the whole document once and mean-pools the token
    embeddings within each chunk's boundaries. With the project's vector-only
    :class:`Embedder` (one vector per text, no token embeddings) we approximate
    that by embedding each sentence and pooling the sentence vectors of every
    ``sentences_per_chunk`` group — so a chunk's embedding still reflects more
    document context than embedding the chunk text in isolation. ``pool_method``
    selects mean or max pooling. ``embed_fn`` is optional: without it, chunk text
    is produced but embeddings are left unset (text-only / testable path).
    """

    name = "late_chunking"

    def __init__(
        self,
        pool_method: str = "mean",
        sentences_per_chunk: int = 3,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        if pool_method not in ("mean", "max"):
            raise ValueError(f"unknown pool_method '{pool_method}'; expected 'mean' or 'max'")
        if sentences_per_chunk < 1:
            raise ValueError("sentences_per_chunk must be >= 1")
        self.pool_method = pool_method
        self.sentences_per_chunk = sentences_per_chunk
        self._embed_fn = embed_fn
        self.config: dict[str, Any] = {
            "pool_method": pool_method,
            "sentences_per_chunk": sentences_per_chunk,
        }

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        sentences = split_sentences(doc.text)
        groups = [
            sentences[i : i + self.sentences_per_chunk]
            for i in range(0, len(sentences), self.sentences_per_chunk)
        ]
        texts = [" ".join(g) for g in groups]
        chunks = build_chunks(doc, texts)
        if self._embed_fn is not None and chunks:
            sent_vecs = self._embed_fn(sentences)
            by_sentence = dict(zip(sentences, sent_vecs, strict=True))
            # Pool the member-sentence vectors for each non-empty (post-strip) chunk.
            for chunk, group in zip(
                chunks, [g for g in groups if " ".join(g).strip()], strict=True
            ):
                chunk.embedding = _pool([by_sentence[s] for s in group], self.pool_method)
        return chunks


# Propositionizer prompt: decompose into atomic, self-contained, decontextualised
# propositions. Adapted from Chen et al. 2023 ("Dense X Retrieval").
_PROPOSITION_PROMPT = (
    "Decompose the following document into a list of clear, simple, self-contained "
    "propositions. Each proposition must:\n"
    "- express a single atomic fact;\n"
    "- be understandable without the rest of the document (resolve pronouns and "
    "references to explicit entities);\n"
    "- stay faithful to the source wording where possible.\n\n"
    "Return ONLY a JSON array of strings, nothing else.\n\n"
    "<document>\n{document}\n</document>"
)


class PropositionalChunker:
    """LLM-based decomposition into atomic propositions (SPEC §7.2).

    Each proposition becomes one chunk. The model call is injectable via
    ``completion_fn`` (prompt -> text) so the chunker is unit-testable offline;
    the default path uses a synchronous Anthropic client (imported lazily).
    Propositions are rewrites, not substrings, so they carry no ``span``.
    """

    name = "propositional"

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 2048,
        completion_fn: CompletionFn | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._completion_fn = completion_fn
        self.config: dict[str, Any] = {"model": model}

    def _default_completion_fn(self) -> CompletionFn:
        def _complete(prompt: str) -> str:
            from anthropic import Anthropic  # type: ignore

            client = Anthropic()
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

        return _complete

    @staticmethod
    def _parse(raw: str) -> list[str]:
        """Parse propositions from model output: JSON array, else one per line."""
        text = raw.strip()
        # Tolerate a ```json fenced block around the array.
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(p).strip() for p in data if str(p).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: newline-delimited, stripping list bullets/numbering.
        lines = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", ln).strip() for ln in text.splitlines()]
        return [ln for ln in lines if ln]

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        if not doc.text.strip():
            return []
        complete = self._completion_fn or self._default_completion_fn()
        propositions = self._parse(complete(_PROPOSITION_PROMPT.format(document=doc.text)))
        return build_chunks(doc, propositions)


__all__ = [
    "RecursiveChunker",
    "MarkdownHeaderChunker",
    "SemanticChunker",
    "SentenceWindowChunker",
    "LateChunkingChunker",
    "PropositionalChunker",
]
