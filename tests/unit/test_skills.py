"""Tests for the skills registry: discover, select, load (SPEC §6.8)."""

from pathlib import Path

import pytest

from common.schemas import Query
from common.types import Skill, SkillManifest
from harness.context import DefaultPacker
from harness.skills import SkillRegistry, keyword_overlap_classify, parse_skill_md

# The shipped seed skills (src/skills/) — repo-root-relative.
SKILLS_ROOT = Path(__file__).resolve().parents[2] / "src" / "skills"


def _write_skill(root: Path, name: str, when: str, body: str, scripts=()):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    front = f"---\nname: {name}\ndescription: {name} desc\nwhen_to_use: {when}\n---\n"
    (skill_dir / "SKILL.md").write_text(front + body, encoding="utf-8")
    for script in scripts:
        (skill_dir / script).write_text("# helper\n", encoding="utf-8")
    return skill_dir


# --- front-matter parsing ---


def test_parse_skill_md_splits_front_matter_and_body():
    front, body = parse_skill_md("---\nname: x\ndescription: hi\n---\n\n# Body\ntext")
    assert front == {"name": "x", "description": "hi"}
    assert body.startswith("# Body")


def test_parse_skill_md_without_front_matter():
    front, body = parse_skill_md("# Just a body")
    assert front == {}
    assert body == "# Just a body"


# --- discovery ---


def test_discover_finds_seed_skills():
    reg = SkillRegistry()
    manifests = reg.discover(SKILLS_ROOT)
    names = {m.name for m in manifests}
    assert {"cite-precisely", "compare-and-contrast"} <= names
    assert all(isinstance(m, SkillManifest) for m in manifests)


def test_discover_records_scripts(tmp_path):
    _write_skill(tmp_path, "with-script", "use me", "# Body", scripts=["run.py"])
    reg = SkillRegistry()
    manifests = reg.discover(tmp_path)
    assert manifests[0].scripts == ["run.py"]


# --- selection ---


@pytest.mark.asyncio
async def test_select_returns_top_k_loaded_skills(tmp_path):
    _write_skill(tmp_path, "compare", "compare and contrast options", "# Compare body")
    _write_skill(tmp_path, "cite", "cite sources precisely", "# Cite body")
    reg = SkillRegistry(classify=keyword_overlap_classify)
    manifests = reg.discover(tmp_path)

    selected = await reg.select(Query(raw="please compare these options"), manifests, k=1)
    assert len(selected) == 1
    assert selected[0].name == "compare"
    assert isinstance(selected[0], Skill)
    assert "Compare body" in selected[0].instructions


@pytest.mark.asyncio
async def test_select_empty_available_returns_empty():
    reg = SkillRegistry(classify=keyword_overlap_classify)
    assert await reg.select(Query(raw="x"), [], k=2) == []


@pytest.mark.asyncio
async def test_select_filters_hallucinated_names(tmp_path):
    _write_skill(tmp_path, "real", "real skill", "# Real")

    async def classify(query, available, k):
        return ["nonexistent", "real"]

    reg = SkillRegistry(classify=classify)
    manifests = reg.discover(tmp_path)
    selected = await reg.select(Query(raw="x"), manifests, k=2)
    assert [s.name for s in selected] == ["real"]


# --- loading ---


def test_load_reads_body_and_scripts(tmp_path):
    _write_skill(tmp_path, "withhelper", "use", "# Instructions here", scripts=["a.py"])
    reg = SkillRegistry()
    reg.discover(tmp_path)
    skill = reg.load("withhelper")
    assert "# Instructions here" in skill.instructions
    assert "a.py" in skill.scripts


def test_load_unknown_raises():
    reg = SkillRegistry()
    with pytest.raises(KeyError):
        reg.load("never-discovered")


# --- packer integration: loaded skills inject into context ---


def test_loaded_skill_injects_into_packer():
    reg = SkillRegistry()
    reg.discover(SKILLS_ROOT)
    skill = reg.load("cite-precisely")
    packed = DefaultPacker().pack(
        system="SYS",
        skills=[skill],
        memory_hits=[],
        retrieval=None,
        scratchpad="",
        messages=[],
        budget_tokens=4000,
    )
    system_text = str(packed[0].content)
    assert "cite-precisely" in system_text
