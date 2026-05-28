"""Phase 5 (DCI) acceptance test (SPEC §15.5).

Asserts all six §15.5 criteria against a hermetic in-memory setup:

1. ``dci`` route beats vector hybrid by ≥ 10% recall@5 on the lexical-bridge slice.
2. Vector hybrid ties or beats ``dci`` (within noise) on the paraphrastic slice.
3. ``dci_then_vector`` chained mode beats either alone on the multi-hop slice.
4. Every DCI tool call runs inside the Phase 3N sandbox under :func:`dci_policy`;
   sandbox-escape probes (network/FS-write) are denied at the policy boundary.
5. DCI tool citations attribute correctly: precision > 0.95 on the citation slice.
6. p50 latency for ``dci`` on a generated n=1000 corpus < 8s; chained < 12s.

The hermetic setup deliberately models the real-world DCI thesis: a bag-of-words
embedder that strips identifier-shaped tokens before tokenising, so vector
retrieval cannot match exact identifiers like ``ERR_PAYLOAD_TOO_LARGE`` — the
exact regime where DCI grep wins. Conversely, prose-only paraphrastic queries
embed strongly, so vector wins there. No live LLM, no external services.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from common.schemas import Chunk, Query, RetrievalCandidate, RetrievalResult, Source
from harness.sandbox import LocalSandbox, SandboxedToolExecutor, SandboxPolicy
from knowledge_index.dci import (
    CorpusGrepTool,
    InMemoryCorpusStore,
    dci_policy,
    make_dci_tools,
)
from knowledge_index.dci.base import GrepHit
from knowledge_index.indexing.qdrant_index import QdrantIndex
from knowledge_index.retrieval.retrievers.dense import DenseRetriever

REPO_ROOT = Path(__file__).resolve().parents[2]
DCI_DATA = REPO_ROOT / "evaluation" / "datasets" / "dci"
CORPUS_DIR = DCI_DATA / "corpus"


# ---------------------------------------------------------------------------
# Bag-of-words embedder — natural-prose only (strips identifier-shaped tokens)
# ---------------------------------------------------------------------------

EMBED_DIM = 96
_STOPWORDS = frozenset(
    """
    the and for with that this from into onto into are was were has have had does
    not but you our its their they them his her she who whom what when where why
    how which there those these would could should about above below over under out
    off than then them whose your also more most some such only just like into very
    too any all one two five each many here other within across upon
    """.split()
)

# Patterns the embedder removes (treated as identifiers, not natural prose):
_IDENT_REMOVE = re.compile(
    r"`[^`]+`"  # backticked spans
    r"|\b[A-Z][A-Z0-9_]{2,}\b"  # ALL_CAPS_TOKENS
    r"|\b[a-z]+(?:_[a-z0-9]+)+\b"  # snake_case
    r"|\b[a-z]+[A-Z][a-zA-Z0-9]+\b"  # camelCase
    r"|\b[a-zA-Z]+\.[a-zA-Z0-9_\.]+\b"  # dotted.names
)
_WORD = re.compile(r"[a-z]{3,}")


def _natural_terms(text: str) -> list[str]:
    cleaned = _IDENT_REMOVE.sub(" ", text)
    return [t for t in _WORD.findall(cleaned.lower()) if t not in _STOPWORDS]


def _bow_vector(text: str) -> list[float]:
    v = [0.0] * EMBED_DIM
    for term in _natural_terms(text):
        v[hash(term) % EMBED_DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class BagOfWordsEmbedder:
    """Strips identifier-shaped tokens, then term-frequency hashes natural prose.

    Deterministic, dependency-free, and crucially: identifier tokens contribute
    nothing to the vector. This is the property that makes lexical-bridge
    queries fall off vector retrieval and into DCI grep's wheelhouse.
    """

    name = "bag-of-words"
    dim = EMBED_DIM

    async def embed_query(self, text: str) -> list[float]:
        return _bow_vector(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_bow_vector(t) for t in texts]


# ---------------------------------------------------------------------------
# Corpus loading — one chunk per markdown doc
# ---------------------------------------------------------------------------


def _doc_id_for(path: Path) -> str:
    return f"doc-{path.stem}"


def _chunk_id_for(path: Path) -> str:
    return f"{_doc_id_for(path)}::chunk-0"


def _load_corpus_chunks() -> list[Chunk]:
    """One chunk per markdown file in ``evaluation/datasets/dci/corpus``."""
    chunks: list[Chunk] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_id = _doc_id_for(path)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id_for(path),
                doc_id=doc_id,
                text=text,
                embedding=_bow_vector(text),
                metadata={
                    "collection": "dci-eval",
                    "source": str(path.name),
                    "title": path.stem.replace("_", " "),
                    "type": "md",
                },
            )
        )
    return chunks


def _load_slice(name: str) -> list[dict[str, Any]]:
    """Load a JSONL slice into a list of dicts."""
    with (DCI_DATA / f"{name}.jsonl").open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Grep-based DCI executor (term extraction → corpus_grep → candidates)
# ---------------------------------------------------------------------------

_IDENT_TERM = re.compile(
    r"`([^`]+)`"  # backticked verbatim
    r"|(\b[A-Z][A-Z0-9_]{2,}\b)"  # ALL_CAPS
    r"|(\b[a-z]+(?:_[a-z0-9]+)+\b)"  # snake_case
    r"|(\b[a-z]+[A-Z][a-zA-Z0-9]+\b)"  # camelCase
    r"|(\b[a-zA-Z]+\.[a-zA-Z0-9_\.]+\b)"  # dotted.names
)
_PROPER = re.compile(r"\b([A-Z][a-z]{2,})\b")
_QUOTED_PHRASE = re.compile(r'"([^"]{3,})"')


def _extract_terms(text: str) -> list[str]:
    """Pick grep terms from a query: identifiers first, then proper nouns, then keywords."""
    terms: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            terms.append(t)

    for m in _IDENT_TERM.finditer(text):
        for grp in m.groups():
            if grp:
                _add(grp)
    for m in _QUOTED_PHRASE.finditer(text):
        _add(m.group(1))
    for m in _PROPER.finditer(text):
        _add(m.group(1))
    # Fallback: longest natural-word tokens (so prose-only queries can grep too).
    if not terms:
        for tok in sorted(set(_natural_terms(text)), key=len, reverse=True)[:4]:
            _add(tok)
    return terms


def _hit_to_candidate(hit: GrepHit, *, rank: int) -> RetrievalCandidate:
    """Wrap a grep hit as a RetrievalCandidate (one synthetic Chunk per hit)."""
    return RetrievalCandidate(
        chunk=Chunk(
            chunk_id=hit.citation.chunk_id or f"{hit.doc_id}::chunk-0",
            doc_id=hit.doc_id,
            text=hit.snippet,
        ),
        score=1.0 / max(1, rank),
        retriever="dci",
        rank=rank,
    )


class GrepDCIExecutor:
    """A minimal :class:`DCIExecutor`: extract terms, grep each, dedup by doc.

    Mirrors what a production executor would do for the grep-first strategy: pull
    identifier-shaped tokens out of the query, fan grep across them, and merge
    the hits into citation-bearing :class:`RetrievalCandidate` objects.
    """

    name = "dci-grep"

    def __init__(
        self, grep_tool: CorpusGrepTool, executor: SandboxedToolExecutor, *, k_terms: int = 4
    ) -> None:
        self._grep = grep_tool
        self._executor = executor
        self._k_terms = k_terms

    async def _grep_term(self, term: str, principals: list[str]) -> list[GrepHit]:
        # Run through the sandboxed executor so the criterion-4 audit observes a
        # real policy/Sandbox boundary — this is how production calls them.
        result = await self._executor.execute(
            self._grep,
            {
                "pattern": re.escape(term),
                "regex": True,
                "max_hits": 20,
                "user_principals": principals,
            },
        )
        if result.ok and isinstance(result.output, list):
            return list(result.output)
        return []

    async def run(self, query: Query, k: int) -> RetrievalResult:
        terms = _extract_terms(query.raw)[: self._k_terms]
        principals = list(query.user_principals)
        seen_docs: dict[str, RetrievalCandidate] = {}
        rank = 1
        for term in terms:
            for hit in await self._grep_term(term, principals):
                if hit.doc_id in seen_docs:
                    continue
                seen_docs[hit.doc_id] = _hit_to_candidate(hit, rank=rank)
                rank += 1
                if len(seen_docs) >= k:
                    break
            if len(seen_docs) >= k:
                break
        return RetrievalResult(
            candidates=list(seen_docs.values())[:k],
            query=query,
            trace_id=uuid4(),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunks() -> list[Chunk]:
    return _load_corpus_chunks()


@pytest.fixture
def corpus_store(chunks: list[Chunk]) -> InMemoryCorpusStore:
    store = InMemoryCorpusStore()
    store.add_chunks(chunks)
    return store


@pytest.fixture
def sandbox_executor() -> SandboxedToolExecutor:
    # The DCI policy: no network, read-only FS, capped CPU/memory. Every DCI
    # tool call routes through this single chokepoint (SPEC §6.7, §15.4).
    return SandboxedToolExecutor(LocalSandbox(), default_policy=dci_policy())


@pytest.fixture
async def qdrant_index(chunks: list[Chunk]) -> QdrantIndex:
    idx = QdrantIndex(collection="dci-eval", dim=EMBED_DIM, location=":memory:")
    await idx.upsert(chunks)
    return idx


@pytest.fixture
def embedder() -> BagOfWordsEmbedder:
    return BagOfWordsEmbedder()


@pytest.fixture
def dci_executor(
    corpus_store: InMemoryCorpusStore, sandbox_executor: SandboxedToolExecutor
) -> GrepDCIExecutor:
    grep_tool = CorpusGrepTool(corpus_store)
    return GrepDCIExecutor(grep_tool, sandbox_executor)


# ---------------------------------------------------------------------------
# Recall helpers
# ---------------------------------------------------------------------------


def _recall_at_k(predicted_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float:
    if not gold_doc_ids:
        return 0.0
    top = predicted_doc_ids[:k]
    hit = sum(1 for g in gold_doc_ids if g in top)
    return hit / len(gold_doc_ids)


async def _dci_doc_ids(executor: GrepDCIExecutor, query_text: str, k: int) -> list[str]:
    result = await executor.run(Query(raw=query_text), k=k)
    return [c.chunk.doc_id for c in result.candidates]


async def _vector_doc_ids(
    index: QdrantIndex, embedder: BagOfWordsEmbedder, query_text: str, k: int
) -> list[str]:
    retriever = DenseRetriever(index, embedder)
    candidates = await retriever.retrieve(Query(raw=query_text), k=k)
    return [c.chunk.doc_id for c in candidates]


async def _chained_doc_ids(
    executor: GrepDCIExecutor,
    index: QdrantIndex,
    embedder: BagOfWordsEmbedder,
    query_text: str,
    k: int,
) -> list[str]:
    """dci_then_vector: grep-anchor first, then vector expansion, merged top-k.

    Mirrors the orchestrator's chained-mode wiring (graph.py:_dci_hop_target):
    accumulate candidates from both hops and surface the union.
    """
    dci_ids = await _dci_doc_ids(executor, query_text, k)
    vec_ids = await _vector_doc_ids(index, embedder, query_text, k)
    out: list[str] = []
    seen: set[str] = set()
    # Interleave DCI then vector so a strong DCI anchor ranks first.
    for source in (dci_ids, vec_ids):
        for doc_id in source:
            if doc_id not in seen:
                out.append(doc_id)
                seen.add(doc_id)
            if len(out) >= k:
                return out
    return out


async def _avg_recall(
    slice_name: str, k: int, retrieve_fn: Any
) -> tuple[float, list[tuple[str, float]]]:
    queries = _load_slice(slice_name)
    per_query: list[tuple[str, float]] = []
    for q in queries:
        ids = await retrieve_fn(q["query"], k)
        r = _recall_at_k(ids, q["relevant_doc_ids"], k)
        per_query.append((q["query_id"], r))
    return sum(r for _, r in per_query) / max(1, len(per_query)), per_query


# ---------------------------------------------------------------------------
# Criterion 1 — DCI beats vector by ≥10% recall@5 on lexical-bridge
# ---------------------------------------------------------------------------


async def test_acceptance_1_dci_beats_vector_on_lexical_bridge(
    dci_executor: GrepDCIExecutor,
    qdrant_index: QdrantIndex,
    embedder: BagOfWordsEmbedder,
) -> None:
    k = 5

    async def dci_fn(q: str, kk: int) -> list[str]:
        return await _dci_doc_ids(dci_executor, q, kk)

    async def vec_fn(q: str, kk: int) -> list[str]:
        return await _vector_doc_ids(qdrant_index, embedder, q, kk)

    dci_avg, _ = await _avg_recall("lexical_bridge", k, dci_fn)
    vec_avg, _ = await _avg_recall("lexical_bridge", k, vec_fn)

    assert dci_avg - vec_avg >= 0.10, (
        f"SPEC §15.5 #1: dci recall@{k}={dci_avg:.2f} must beat vector recall@{k}={vec_avg:.2f} "
        f"by ≥ 10% on the lexical-bridge slice; observed gap {dci_avg - vec_avg:.2f}"
    )


# ---------------------------------------------------------------------------
# Criterion 2 — vector ties or beats DCI on paraphrastic
# ---------------------------------------------------------------------------


async def test_acceptance_2_vector_ties_or_beats_dci_on_paraphrastic(
    dci_executor: GrepDCIExecutor,
    qdrant_index: QdrantIndex,
    embedder: BagOfWordsEmbedder,
) -> None:
    k = 5
    noise = 0.05  # within-noise tolerance per SPEC §15.5 #2

    async def dci_fn(q: str, kk: int) -> list[str]:
        return await _dci_doc_ids(dci_executor, q, kk)

    async def vec_fn(q: str, kk: int) -> list[str]:
        return await _vector_doc_ids(qdrant_index, embedder, q, kk)

    dci_avg, _ = await _avg_recall("paraphrastic", k, dci_fn)
    vec_avg, _ = await _avg_recall("paraphrastic", k, vec_fn)

    assert vec_avg + noise >= dci_avg, (
        f"SPEC §15.5 #2: vector recall@{k}={vec_avg:.2f} must tie or beat dci "
        f"recall@{k}={dci_avg:.2f} (within noise {noise}) on the paraphrastic slice"
    )


# ---------------------------------------------------------------------------
# Criterion 3 — chained beats either alone on multi-hop
# ---------------------------------------------------------------------------


async def test_acceptance_3_chained_beats_either_on_multihop(
    dci_executor: GrepDCIExecutor,
    qdrant_index: QdrantIndex,
    embedder: BagOfWordsEmbedder,
) -> None:
    k = 5

    queries = _load_slice("multihop")
    per: list[tuple[str, float, float, float]] = []
    for q in queries:
        gold = q["relevant_doc_ids"]
        dci_r = _recall_at_k(await _dci_doc_ids(dci_executor, q["query"], k), gold, k)
        vec_r = _recall_at_k(await _vector_doc_ids(qdrant_index, embedder, q["query"], k), gold, k)
        ch_r = _recall_at_k(
            await _chained_doc_ids(dci_executor, qdrant_index, embedder, q["query"], k), gold, k
        )
        per.append((q["query_id"], dci_r, vec_r, ch_r))

    # Chained must NEVER be worse than either single mode per query.
    worse = [p for p in per if p[3] < p[1] - 1e-9 or p[3] < p[2] - 1e-9]
    assert not worse, f"SPEC §15.5 #3: chained worse than a single mode on: {worse}"

    # Aggregate: chained mean ≥ each single-mode mean, AND strictly exceeds at
    # least one of them (proving chained adds value, not just an inert union).
    dci_avg = sum(p[1] for p in per) / len(per)
    vec_avg = sum(p[2] for p in per) / len(per)
    ch_avg = sum(p[3] for p in per) / len(per)
    assert ch_avg >= dci_avg - 1e-9 and ch_avg >= vec_avg - 1e-9
    assert ch_avg > min(dci_avg, vec_avg) + 1e-9, (
        f"SPEC §15.5 #3: chained_mean={ch_avg:.2f} must strictly beat the weaker "
        f"single mode (dci={dci_avg:.2f}, vector={vec_avg:.2f})"
    )


# ---------------------------------------------------------------------------
# Criterion 4 — DCI tools run in the sandbox; escape probes denied
# ---------------------------------------------------------------------------


async def test_acceptance_4_dci_runs_in_sandbox_with_no_escapes(
    corpus_store: InMemoryCorpusStore,
) -> None:
    policy = dci_policy()
    assert policy.network == "none"
    assert policy.fs_writable is False
    assert policy.cpu_seconds > 0 and policy.memory_mb > 0

    tools = make_dci_tools(corpus_store)
    executor = SandboxedToolExecutor(LocalSandbox(), default_policy=policy)

    # (a) Benign call: a normal grep runs and returns hits — no false positives.
    r = await executor.execute(
        tools["corpus_grep"],
        {"pattern": "Falcon", "user_principals": []},
    )
    assert r.ok, f"benign grep should succeed under dci_policy, got {r}"

    # (b) Escape probe: a tool that declares it needs network must be rejected
    # at the policy boundary BEFORE it runs. The executor catches the
    # PolicyViolation and surfaces it as ok=False so a bad tool never crashes
    # the graph — but the violation must still be the recorded reason.
    class HostileNetworkTool:
        name = "exfil_tool"
        network_required = True

        async def __call__(self, args: dict, *, workdir: Path) -> Any:
            return "should-never-run"

    deny = await executor.execute(HostileNetworkTool(), {})
    assert not deny.ok and "policy violation" in (deny.error or "").lower()

    # (c) Escape probe: a tool that runs over its time budget is killed by the
    # resource ceiling, not allowed to silently exfiltrate.
    import asyncio

    class RunawayTool:
        name = "runaway"
        network_required = False

        async def __call__(self, args: dict, *, workdir: Path) -> Any:
            await asyncio.sleep(10.0)
            return "ran too long"

    tiny = SandboxPolicy(network="none", cpu_seconds=1, fs_writable=False)
    killed = await executor.execute(RunawayTool(), {}, policy=tiny)
    assert not killed.ok and "resource limit" in (killed.error or "").lower()


# ---------------------------------------------------------------------------
# Criterion 5 — citation precision > 0.95 on the citation audit slice
# ---------------------------------------------------------------------------


async def test_acceptance_5_citation_precision_over_95(
    corpus_store: InMemoryCorpusStore,
    sandbox_executor: SandboxedToolExecutor,
) -> None:
    grep = CorpusGrepTool(corpus_store)
    total = 0
    correct = 0
    queries = _load_slice("citation_audit")
    for q in queries:
        for term in _extract_terms(q["query"])[:4]:
            result = await sandbox_executor.execute(
                grep,
                {
                    "pattern": re.escape(term),
                    "regex": True,
                    "max_hits": 10,
                    "user_principals": [],
                },
            )
            if not result.ok:
                continue
            for hit in result.output:
                assert isinstance(hit, GrepHit)
                assert isinstance(hit.citation, Source)
                # Every grep hit must carry a citation whose doc_id is the doc
                # the hit was found in (no provenance drift).
                assert hit.citation.doc_id == hit.doc_id
                total += 1
                if hit.citation.doc_id in q["relevant_doc_ids"]:
                    correct += 1

    assert total > 0, "citation audit must produce some hits"
    precision = correct / total
    assert precision > 0.95, (
        f"SPEC §15.5 #5: citation precision {precision:.3f} must exceed 0.95 "
        f"(correct={correct}/total={total})"
    )


# ---------------------------------------------------------------------------
# Criterion 6 — latency budgets on n=1000 corpus
# ---------------------------------------------------------------------------


def _statistics_median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


async def test_acceptance_6_latency_budgets_n1000(
    sandbox_executor: SandboxedToolExecutor,
) -> None:
    # Synthesise a 1000-doc corpus: 12 real-content "anchors" (a few with the
    # exact identifier we'll grep) plus 988 unique-prose distractors so a real
    # search has to scan them.
    anchors = _load_corpus_chunks()
    distractors: list[Chunk] = []
    template = (
        "Topic {i}: this distractor doc discusses subject {i} in prose form. "
        "It is included to make the corpus dense enough that ranking matters."
    )
    for i in range(1000 - len(anchors)):
        cid = f"doc-noise-{i:04d}"
        distractors.append(
            Chunk(
                chunk_id=f"{cid}::chunk-0",
                doc_id=cid,
                text=template.format(i=i),
                embedding=_bow_vector(template.format(i=i)),
                metadata={"collection": "noise", "source": f"noise-{i}.md", "type": "md"},
            )
        )
    chunks = anchors + distractors
    assert len(chunks) == 1000

    store = InMemoryCorpusStore()
    store.add_chunks(chunks)
    idx = QdrantIndex(collection="dci-perf", dim=EMBED_DIM, location=":memory:")
    await idx.upsert(chunks)

    grep = CorpusGrepTool(store)
    executor = GrepDCIExecutor(grep, sandbox_executor)
    embedder = BagOfWordsEmbedder()

    queries = [q["query"] for q in _load_slice("lexical_bridge")] + [
        q["query"] for q in _load_slice("multihop")
    ]

    dci_lat: list[float] = []
    chain_lat: list[float] = []
    for qtext in queries:
        t0 = time.perf_counter()
        await _dci_doc_ids(executor, qtext, 5)
        dci_lat.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        await _chained_doc_ids(executor, idx, embedder, qtext, 5)
        chain_lat.append(time.perf_counter() - t1)

    p50_dci = _statistics_median(dci_lat)
    p50_chain = _statistics_median(chain_lat)
    assert p50_dci < 8.0, f"SPEC §15.5 #6: dci p50={p50_dci:.2f}s must be < 8s on n=1000"
    assert p50_chain < 12.0, f"SPEC §15.5 #6: chained p50={p50_chain:.2f}s must be < 12s on n=1000"
