"""Skills: loadable instruction packs with LLM-based selection (SPEC §6.8)."""

from common.types import Skill, SkillManifest
from harness.skills.registry import (
    DEFAULT_SELECT_MODEL,
    SKILL_FILE,
    ClassifyFn,
    SkillRegistry,
    keyword_overlap_classify,
    parse_skill_md,
)

__all__ = [
    "Skill",
    "SkillManifest",
    "SkillRegistry",
    "ClassifyFn",
    "SKILL_FILE",
    "DEFAULT_SELECT_MODEL",
    "parse_skill_md",
    "keyword_overlap_classify",
]
