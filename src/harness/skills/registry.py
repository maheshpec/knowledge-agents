"""Skill discovery, selection, and loading (SPEC §6.8).

A *skill* is a directory containing a ``SKILL.md`` (YAML front-matter + markdown
body) and optional helper scripts. The registry:

- :meth:`discover` walks a tree for ``SKILL.md`` files and parses their
  front-matter into cheap :class:`SkillManifest` records (no body read);
- :meth:`select` asks a fast LLM classifier which skills fit the current query,
  returning the top-k loaded :class:`Skill` objects (the selector is injectable,
  so selection runs offline under test);
- :meth:`load` reads a skill's full instructions + indexes its helper scripts.

Selected skills are injected into context by the packer (SPEC §6.6), which renders
``Skill.name`` + ``Skill.instructions``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import yaml

from common.errors import ConfigError
from common.schemas import Query
from common.types import Skill, SkillManifest
from harness.observability.logging import get_logger
from harness.observability.tracing import traced

_log = get_logger("harness.skills")

SKILL_FILE = "SKILL.md"
DEFAULT_SELECT_MODEL = "claude-haiku-4-5-20251001"

# A classifier maps (query_text, available manifests, k) -> ordered skill names.
ClassifyFn = Callable[[str, list[SkillManifest], int], Awaitable[list[str]]]


def parse_skill_md(text: str) -> tuple[dict, str]:
    """Split a ``SKILL.md`` into (front-matter dict, body).

    Front-matter is a leading ``---``-fenced YAML block; everything after the
    closing fence is the instruction body. A file with no front-matter yields an
    empty dict and the whole text as body.
    """
    if text.lstrip().startswith("---"):
        stripped = text.lstrip()
        parts = stripped.split("---", 2)
        if len(parts) == 3:
            front = yaml.safe_load(parts[1]) or {}
            if not isinstance(front, dict):
                raise ConfigError("SKILL.md front-matter must be a YAML mapping")
            return front, parts[2].lstrip("\n")
    return {}, text


def keyword_overlap_classify(
    query: str, available: list[SkillManifest], k: int
) -> Awaitable[list[str]]:
    """Deterministic, LLM-free selector: rank by query/(desc+when_to_use) word overlap.

    Returned as an awaitable so it is a drop-in for the async classifier contract;
    used as the offline default and in tests.
    """

    async def _run() -> list[str]:
        terms = {t.lower() for t in query.split()}

        def score(m: SkillManifest) -> int:
            haystack = f"{m.name} {m.description} {m.when_to_use}".lower()
            return sum(1 for t in terms if t in haystack)

        ranked = sorted(available, key=score, reverse=True)
        return [m.name for m in ranked[:k]]

    return _run()


class SkillRegistry:
    """Discover, select, and load loadable skills (SPEC §6.8)."""

    def __init__(self, classify: ClassifyFn | None = None, *, model: str = DEFAULT_SELECT_MODEL):
        self._classify = classify
        self._model = model
        self._manifests: dict[str, SkillManifest] = {}

    @traced(span_name="skills.discover")
    def discover(self, root: Path) -> list[SkillManifest]:
        """Walk ``root`` for ``SKILL.md`` files; parse + cache their manifests."""
        root = Path(root)
        manifests: list[SkillManifest] = []
        for skill_md in sorted(root.rglob(SKILL_FILE)):
            front, _ = parse_skill_md(skill_md.read_text(encoding="utf-8"))
            skill_dir = skill_md.parent
            scripts = sorted(
                p.name for p in skill_dir.iterdir() if p.is_file() and p.name != SKILL_FILE
            )
            manifest = SkillManifest(
                name=front.get("name", skill_dir.name),
                description=front.get("description", ""),
                when_to_use=front.get("when_to_use", ""),
                path=str(skill_dir),
                scripts=scripts,
            )
            self._manifests[manifest.name] = manifest
            manifests.append(manifest)
        _log.info("skills.discover", root=str(root), found=len(manifests))
        return manifests

    @traced(span_name="skills.select")
    async def select(self, query: Query, available: list[SkillManifest], k: int = 2) -> list[Skill]:
        """Return the top-k skills the classifier judges relevant to ``query``."""
        if not available or k <= 0:
            return []
        classify = self._classify or self._default_classify
        names = await classify(query.raw, available, k)
        available_names = {m.name for m in available}
        selected = [n for n in names if n in available_names][:k]
        _log.info("skills.select", k=k, selected=selected)
        return [self.load(n) for n in selected]

    @traced(span_name="skills.load")
    def load(self, name: str) -> Skill:
        """Load a discovered skill's full instructions + helper script index."""
        manifest = self._manifests.get(name)
        if manifest is None or manifest.path is None:
            raise KeyError(f"unknown skill '{name}'; run discover() first")
        skill_dir = Path(manifest.path)
        _, body = parse_skill_md((skill_dir / SKILL_FILE).read_text(encoding="utf-8"))
        scripts = {s: str(skill_dir / s) for s in manifest.scripts}
        return Skill(
            name=manifest.name,
            instructions=body.strip(),
            description=manifest.description,
            when_to_use=manifest.when_to_use,
            path=manifest.path,
            scripts=scripts,
        )

    async def _default_classify(
        self, query: str, available: list[SkillManifest], k: int
    ) -> list[str]:
        """LLM selector (Haiku): pick the k most relevant skill names for the query."""
        from langchain_anthropic import ChatAnthropic

        from common.settings import get_settings

        catalog = "\n".join(f"- {m.name}: {m.when_to_use or m.description}" for m in available)
        prompt = (
            f"Available skills:\n{catalog}\n\n"
            f"For the user query below, list up to {k} skill names (exact names, one per "
            f"line) that are most relevant. If none are relevant, output nothing.\n\n"
            f"Query: {query}"
        )
        init_kwargs: dict = {
            "model": self._model,
            "api_key": get_settings().anthropic_api_key,
            "max_tokens": 128,
        }
        llm = ChatAnthropic(**init_kwargs)
        response = await llm.ainvoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        lines = [ln.strip("- \t") for ln in content.splitlines() if ln.strip()]
        return lines[:k]


__all__ = [
    "SKILL_FILE",
    "DEFAULT_SELECT_MODEL",
    "ClassifyFn",
    "SkillRegistry",
    "parse_skill_md",
    "keyword_overlap_classify",
]
