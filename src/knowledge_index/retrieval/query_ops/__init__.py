"""Query operations: Rewriter (Phase 1) + Phase 3 stubs (SPEC §7.6.2)."""

from knowledge_index.retrieval.query_ops.base import (
    DEFAULT_QUERY_OP_MODEL,
    CompleteFn,
    QueryOp,
    apply_query_ops,
    default_completer,
)
from knowledge_index.retrieval.query_ops.rewriter import REWRITE_PROMPT, Rewriter
from knowledge_index.retrieval.query_ops.stubs import Decomposer, HyDEExpander, Stepback

__all__ = [
    "CompleteFn",
    "DEFAULT_QUERY_OP_MODEL",
    "QueryOp",
    "apply_query_ops",
    "default_completer",
    "REWRITE_PROMPT",
    "Rewriter",
    "HyDEExpander",
    "Decomposer",
    "Stepback",
]
