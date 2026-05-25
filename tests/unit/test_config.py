"""Tests for Hydra config plumbing (SPEC §4)."""

import pytest

from common.config import load_config, to_container
from common.errors import ConfigError


def test_load_default_config():
    cfg = load_config("default")
    assert cfg.llm.generation_model == "claude-sonnet-4-6"
    assert cfg.index.retrieval.fusion.rrf_k == 60


def test_load_config_with_override():
    cfg = load_config("default", overrides=["budget.max_usd=5"])
    assert cfg.budget.max_usd == 5


def test_to_container_resolves_to_dict():
    cfg = load_config("eval")
    container = to_container(cfg)
    assert isinstance(container, dict)
    assert container["datasets"]["frozen"]["size"] == 1000


def test_missing_config_raises():
    with pytest.raises(ConfigError):
        load_config("does_not_exist")
