"""scripts/eval_run.py — evaluation harness scaffold (SPEC §9, epic ka-2vc deliverable 7).

Phase 1 scaffold: loads a tiny seed dataset (10 queries), runs each through the
full pipeline, and reports cost / latency / citation metrics. Real metrics
(recall@k, nDCG, ragas faithfulness) and the dev/frozen datasets land in Phase 2
(§9.1–§9.3); this establishes the runner shape and output contract.

Usage::

    uv run scripts/eval_run.py --dataset seed --collection main
    uv run scripts/eval_run.py --offline      # in-memory corpus, no API keys
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any

# A tiny built-in seed dataset (Phase 2 replaces this with evaluation/datasets/).
SEED_QUERIES: list[dict[str, str]] = [
    {"query_id": f"seed-{i}", "query": q}
    for i, q in enumerate(
        [
            "What does the mitochondrion do?",
            "How does photosynthesis work?",
            "How tall is Mount Everest?",
            "What is the formula of water?",
            "What does insulin regulate?",
            "What is TCP?",
            "Who wrote Hamlet?",
            "What is Python?",
            "When was Rome founded?",
            "What is gravity?",
        ]
    )
]

_OFFLINE_CORPUS = {
    "mitochondria": "The mitochondrion is the powerhouse of the cell, producing ATP.",
    "photosynthesis": "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen.",
    "everest": "Mount Everest is the highest mountain at 8849 metres.",
    "water": "Water has the chemical formula H2O.",
    "insulin": "Insulin is a hormone that regulates blood glucose.",
    "tcp": "TCP is a reliable, connection-oriented transport protocol.",
    "shakespeare": "William Shakespeare wrote Hamlet and Macbeth.",
    "python": "Python is a high-level programming language.",
    "rome": "Rome was founded in 753 BC.",
    "gravity": "Gravity accelerates objects on Earth at 9.8 m/s^2.",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the seed evaluation set through the pipeline.")
    p.add_argument("--dataset", default="seed", help="Dataset name (Phase 1: only 'seed').")
    p.add_argument("--collection", default="main", help="Qdrant collection (non-offline mode).")
    p.add_argument("--strictness", default="strict", choices=["strict", "loose", "off"])
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use an in-memory corpus + hash embedder + stub draft (no API keys).",
    )
    p.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    return p.parse_args(argv)


async def _build_offline_app(tmp_dir: str):  # type: ignore[no-untyped-def]
    from harness.citation import CitationEnforcer, CitedDraft, CitedSegment
    from harness.context import DefaultPacker
    from harness.orchestrator import OrchestratorDeps, build_orchestrator
    from harness.planning import ReactPlanner
    from knowledge_index.chunking import RecursiveChunker
    from knowledge_index.embedding import HashEmbedder
    from knowledge_index.enrichment import TitleEnricher
    from knowledge_index.indexing import QdrantIndex
    from knowledge_index.pipeline import IngestionPipeline
    from knowledge_index.retrieval import (
        DenseRetriever,
        HybridPipeline,
        MMRDiversifier,
        NullReranker,
        RRFFuser,
        SparseBM25Retriever,
    )

    dim = 96
    index = QdrantIndex("eval-offline", dim=dim, location=":memory:")
    embedder = HashEmbedder(dim=dim)
    ingest = IngestionPipeline(
        chunker=RecursiveChunker(chunk_size=300, chunk_overlap=40),
        enricher=TitleEnricher(),
        embedder=embedder,
        index=index,
    )
    from pathlib import Path

    for name, text in _OFFLINE_CORPUS.items():
        Path(tmp_dir, f"{name}.md").write_text(f"# {name.title()}\n\n{text}\n")
    await ingest.ingest_dir(tmp_dir, acl=[])

    async def _draft(question, candidates):
        if not candidates:
            return CitedDraft(refused=True, refusal_reason="no evidence")
        top = candidates[0].chunk
        return CitedDraft(segments=[CitedSegment(text=top.text, citation_ids=[top.chunk_id])])

    pipeline = HybridPipeline(
        retrievers=[DenseRetriever(index, embedder), SparseBM25Retriever(index)],
        reranker=NullReranker(),
        fuser=RRFFuser(),
        post_processors=[MMRDiversifier(embedder.embed_query, lambda_=0.5)],
    )
    deps = OrchestratorDeps(
        pipeline=pipeline,
        enforcer=CitationEnforcer(draft_fn=_draft),
        packer=DefaultPacker(),
        planner=ReactPlanner(),
    )
    return build_orchestrator(deps)


async def run_seed_eval(args: argparse.Namespace) -> dict[str, Any]:
    from harness import answer

    app = None
    if args.offline:
        import tempfile

        app = await _build_offline_app(tempfile.mkdtemp())

    per_query: list[dict[str, Any]] = []
    for item in SEED_QUERIES:
        start = time.perf_counter()
        opts: dict[str, Any] = {"strictness": args.strictness}
        if app is not None:
            opts["app"] = app
        else:
            opts["collection"] = args.collection
        result = await answer(item["query"], **opts)
        latency_ms = (time.perf_counter() - start) * 1000.0
        per_query.append(
            {
                "query_id": item["query_id"],
                "latency_ms": round(latency_ms, 2),
                "cost_usd": round(result.cost, 6),
                "num_citations": len(result.citations),
                "cited": bool(result.citations),
            }
        )

    latencies = [q["latency_ms"] for q in per_query]
    return {
        "dataset": args.dataset,
        "n": len(per_query),
        "citation_rate": sum(q["cited"] for q in per_query) / len(per_query),
        "total_cost_usd": round(sum(q["cost_usd"] for q in per_query), 6),
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2),
        "per_query": per_query,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = asyncio.run(run_seed_eval(args))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"[{report['dataset']}] n={report['n']} "
            f"citation_rate={report['citation_rate']:.2f} "
            f"cost=${report['total_cost_usd']:.4f} "
            f"p50={report['latency_p50_ms']}ms p95={report['latency_p95_ms']}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
