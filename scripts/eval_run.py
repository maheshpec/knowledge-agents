"""scripts/eval_run.py — evaluation harness (SPEC §9, epic ka-5ps deliverable 5).

Runs a named dataset (``dev`` / ``rotating`` / ``frozen``) through the retrieval +
orchestration pipeline, scores it with the full metric suite (retrieval, e2e,
operational), and emits a complete :class:`EvalReport` as JSON.

Usage::

    uv run scripts/eval_run.py --dataset dev --pipeline configs/default.yaml
    uv run scripts/eval_run.py --dataset dev --offline --json report.json
    uv run scripts/eval_run.py --dataset dev --smoke 20            # PR smoke subset
    uv run scripts/eval_run.py --dataset dev --gate-baseline main.json  # CI gate

Without API keys the runner auto-selects ``--offline`` mode: an in-memory corpus
(the seed fixture) + hash embedder + a stub citing draft, so it runs key-free in
CI. Pass ``--online`` to force the real Anthropic/Voyage/Qdrant stack.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from common.config import to_container
from common.errors import ConfigError
from evaluation.datasets import corpus_dir, load_dataset
from evaluation.metrics import default_metrics
from evaluation.runners import EvalReport, EvalRunner, OrchestratorPipelineRunner
from self_improvement.registry.pipeline_config import PipelineConfig

DEFAULT_OFFLINE_DIM = 96


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an evaluation dataset through the pipeline.")
    p.add_argument("--dataset", default="dev", help="Dataset split: dev | rotating | frozen.")
    p.add_argument("--pipeline", default="configs/default.yaml", help="Pipeline config (lineage).")
    p.add_argument("--collection", default="main", help="Qdrant collection (online mode).")
    p.add_argument("--strictness", default="strict", choices=["strict", "loose", "off"])
    p.add_argument("--smoke", type=int, default=0, help="Run only the first N queries (PR smoke).")
    p.add_argument("--concurrency", type=int, default=8, help="Max concurrent queries.")
    p.add_argument("--offline", action="store_true", help="Force offline (key-free) mode.")
    p.add_argument("--online", action="store_true", help="Force online (real services) mode.")
    p.add_argument("--allow-frozen", action="store_true", help="Permit loading the frozen split.")
    p.add_argument("--json", nargs="?", const="-", help="Emit JSON (optionally to a file path).")
    p.add_argument("--assert-recall10", type=float, default=None, help="Fail if recall@10 below.")
    p.add_argument(
        "--gate-baseline", default=None, help="Baseline report JSON for regression gate."
    )
    p.add_argument(
        "--gate-max-regression",
        type=float,
        default=0.05,
        help="Max fractional recall@10 drop vs baseline before failing (default 0.05).",
    )
    return p.parse_args(argv)


def _has_keys() -> bool:
    try:
        from common.settings import get_settings

        s = get_settings()
        return bool(getattr(s, "anthropic_api_key", None) and getattr(s, "voyage_api_key", None))
    except Exception:
        return False


def _pipeline_config(pipeline_path: str) -> PipelineConfig:
    """Read the ``index.retrieval`` block of a Hydra config into a PipelineConfig (lineage)."""
    try:
        from common.config import load_config

        name = Path(pipeline_path).stem
        cfg = to_container(load_config(name))
        retr = cfg.get("index", {}).get("retrieval", {})  # type: ignore[union-attr]
        return PipelineConfig(
            retrievers=list(retr.get("retrievers", ["dense", "sparse_bm25"])),
            fusion=retr.get("fusion", {}).get("name", "rrf"),
            rrf_k=int(retr.get("fusion", {}).get("rrf_k", 60)),
            reranker=retr.get("reranker", {}).get("name", "cohere_rerank_3"),
            reranker_top_k=int(retr.get("reranker", {}).get("top_k", 10)),
            post_processors=list(retr.get("post_processors", ["mmr", "parent_expander"])),
            query_ops=list(retr.get("query_ops", ["rewrite"])),
        )
    except Exception:
        return PipelineConfig()


async def _build_offline_app(corpus: Path) -> Any:
    """In-memory app: hash embedder + BM25 + stub citing draft (no API keys).

    Ingests the fixture ``corpus`` with the same chunker config the seed dataset
    was generated against, so retrieved chunk/doc ids match the gold labels.
    """
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

    dim = DEFAULT_OFFLINE_DIM
    index = QdrantIndex("eval-offline", dim=dim, location=":memory:")
    embedder = HashEmbedder(dim=dim)
    ingest = IngestionPipeline(
        chunker=RecursiveChunker(chunk_size=500, chunk_overlap=75),
        enricher=TitleEnricher(),
        embedder=embedder,
        index=index,
    )
    if not corpus.exists():
        raise ConfigError(f"offline corpus not found: {corpus} (run scripts/gen_seed_dataset.py)")
    await ingest.ingest_dir(str(corpus), acl=[])

    async def _draft(question: str, candidates: list) -> CitedDraft:  # type: ignore[type-arg]
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


async def run_eval(args: argparse.Namespace) -> EvalReport:
    offline = args.offline or (not args.online and not _has_keys())
    dataset = load_dataset(args.dataset, allow_frozen=args.allow_frozen)
    if args.smoke > 0:
        dataset = dataset.subset(args.smoke)

    if offline:
        app = await _build_offline_app(corpus_dir())
    else:
        from harness import build_default_deps
        from harness.orchestrator import build_orchestrator

        app = build_orchestrator(build_default_deps(collection=args.collection))

    runner = OrchestratorPipelineRunner(app, strictness=args.strictness)
    eval_runner = EvalRunner(runner, concurrency=args.concurrency, k=20)
    return await eval_runner.run(_pipeline_config(args.pipeline), dataset, default_metrics())


def _check_gates(report: EvalReport, args: argparse.Namespace) -> int:
    """Apply CI gates; return a process exit code (0 = pass)."""
    code = 0
    recall10 = report.recall_at(10)
    if args.assert_recall10 is not None and recall10 < args.assert_recall10:
        print(
            f"FAIL: recall@10={recall10:.3f} < required {args.assert_recall10:.3f}",
            file=sys.stderr,
        )
        code = 1
    if args.gate_baseline:
        baseline = EvalReport.model_validate_json(Path(args.gate_baseline).read_text())
        base_recall = baseline.recall_at(10)
        floor = base_recall * (1.0 - args.gate_max_regression)
        if recall10 < floor:
            print(
                f"FAIL: recall@10 regressed: {recall10:.3f} < {floor:.3f} "
                f"(baseline {base_recall:.3f}, max drop {args.gate_max_regression:.0%})",
                file=sys.stderr,
            )
            code = 1
        else:
            print(f"OK: recall@10={recall10:.3f} vs baseline {base_recall:.3f} (floor {floor:.3f})")
    return code


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = asyncio.run(run_eval(args))

    if args.json is not None:
        payload = json.dumps(report.model_dump(), indent=2)
        if args.json == "-":
            print(payload)
        else:
            Path(args.json).write_text(payload)
            print(f"wrote report to {args.json}")
    else:
        agg = report.aggregated
        ordered = " ".join(f"{k}={agg[k]:.3f}" for k in sorted(agg))
        print(f"[{report.dataset}] n={report.n} duration={report.duration_s}s")
        print(ordered)

    return _check_gates(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
