"""Config-diff generation for self-improvement PRs (SPEC §8.4).

An accepted candidate is a :class:`PipelineConfig`; landing it means updating the
``index.retrieval`` block of ``configs/default.yaml``. These helpers map the
config onto that block, re-serialize the file, and produce a unified diff for the
PR body — without ever mutating a file on disk (the generator decides where the
updated text is written).

Round-trip note: only the ``index.retrieval`` subtree is touched; every other key
(and the relative ordering of retrieval keys that already exist) is preserved so
the diff is minimal and reviewable.
"""

from __future__ import annotations

import copy
import difflib
from typing import Any

import yaml

from self_improvement.registry.pipeline_config import PipelineConfig


def pipeline_config_to_retrieval_block(config: PipelineConfig) -> dict[str, Any]:
    """Render a :class:`PipelineConfig` into the ``index.retrieval`` YAML shape.

    Mirrors the structure in ``configs/default.yaml``: ``fusion`` and ``reranker``
    are nested mappings (name + tuned param), the rest are scalars/lists. The
    evolutionary loop also tunes ``mmr_lambda``, surfaced as a sibling key so it
    round-trips through the config.
    """
    return {
        "retrievers": list(config.retrievers),
        "fusion": {"name": config.fusion, "rrf_k": config.rrf_k},
        "reranker": {"name": config.reranker, "top_k": config.reranker_top_k},
        "post_processors": list(config.post_processors),
        "mmr_lambda": config.mmr_lambda,
        "query_ops": list(config.query_ops),
    }


def apply_pipeline_config(current: dict[str, Any], config: PipelineConfig) -> dict[str, Any]:
    """Return a deep copy of ``current`` with ``index.retrieval`` set from ``config``.

    Does not mutate the input. Creates the ``index`` / ``index.retrieval`` nodes
    if the (stub) config lacks them, so this works against a minimal file too.
    """
    updated = copy.deepcopy(current)
    index = updated.setdefault("index", {})
    if not isinstance(index, dict):
        raise TypeError("config 'index' key is not a mapping")
    index["retrieval"] = pipeline_config_to_retrieval_block(config)
    return updated


def _dump(data: dict[str, Any]) -> str:
    """Stable YAML serialization: preserve key order, block style, no aliases."""
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def _load(text: str) -> dict[str, Any]:
    current = yaml.safe_load(text) or {}
    if not isinstance(current, dict):
        raise TypeError("default config must parse to a mapping")
    return current


def normalize_config_text(text: str) -> str:
    """Re-serialize ``text`` through the canonical dumper (formatting-only pass).

    The diff is taken against this normalized form, so it reflects *semantic*
    changes (values/keys) rather than incidental formatting differences between
    the hand-edited file and the generator's output — which is exactly what gets
    written to the branch.
    """
    return _dump(_load(text))


def render_config_update(current_text: str, config: PipelineConfig) -> str:
    """Parse ``current_text``, apply ``config``, and return the updated YAML text."""
    return _dump(apply_pipeline_config(_load(current_text), config))


def unified_config_diff(
    current_text: str,
    updated_text: str,
    *,
    path: str = "configs/default.yaml",
) -> str:
    """Unified diff between two YAML texts, labelled a/<path> → b/<path>.

    Returns an empty string when the texts are identical (a no-op candidate),
    which the generator treats as "nothing to PR".
    """
    diff = difflib.unified_diff(
        current_text.splitlines(keepends=True),
        updated_text.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff)


def config_diff_for_candidate(
    current_text: str,
    config: PipelineConfig,
    *,
    path: str = "configs/default.yaml",
) -> tuple[str, str]:
    """Produce ``(updated_text, unified_diff)`` for a candidate config.

    The diff is normalized-current → updated, so it is empty when the candidate is
    semantically identical to the current config (a no-op the generator refuses)
    and otherwise shows only the real changes.
    """
    updated_text = render_config_update(current_text, config)
    diff = unified_config_diff(normalize_config_text(current_text), updated_text, path=path)
    return updated_text, diff


__all__ = [
    "pipeline_config_to_retrieval_block",
    "apply_pipeline_config",
    "normalize_config_text",
    "render_config_update",
    "unified_config_diff",
    "config_diff_for_candidate",
]
