"""Query operations: Rewriter, HyDE, decomposition, step-back (SPEC §7.6.2)."""

from knowledge_index.retrieval.query_ops.base import (
    DEFAULT_QUERY_OP_MODEL,
    CompleteFn,
    QueryOp,
    apply_query_ops,
    default_completer,
)
from knowledge_index.retrieval.query_ops.expanders import (
    DECOMPOSE_PROMPT,
    HYDE_PROMPT,
    STEPBACK_PROMPT,
    Decomposer,
    HyDEExpander,
    Stepback,
)
from knowledge_index.retrieval.query_ops.rewriter import REWRITE_PROMPT, Rewriter

__all__ = [
    "CompleteFn",
    "DEFAULT_QUERY_OP_MODEL",
    "QueryOp",
    "apply_query_ops",
    "default_completer",
    "REWRITE_PROMPT",
    "Rewriter",
    "HYDE_PROMPT",
    "DECOMPOSE_PROMPT",
    "STEPBACK_PROMPT",
    "HyDEExpander",
    "Decomposer",
    "Stepback",
]
