"""Hydra config plumbing (SPEC §4, §8.1).

Thin helpers around Hydra's Compose API so scripts and tests can load the
composable YAML configs without a full ``@hydra.main`` entry point. Scripts that
want CLI overrides can still use ``@hydra.main``; these helpers cover
programmatic loads (tests, the eval runner, the evolutionary loop).
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from common.errors import ConfigError

# Repo-root-relative configs directory (…/knowledge_agents/configs).
CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(
    config_name: str = "default",
    *,
    overrides: list[str] | None = None,
    configs_dir: Path | None = None,
) -> DictConfig:
    """Compose a Hydra config by name with optional dotlist overrides.

    Example::

        cfg = load_config("default", overrides=["budget.max_usd=5"])
    """
    cfg_dir = (configs_dir or CONFIGS_DIR).resolve()
    if not (cfg_dir / f"{config_name}.yaml").exists():
        raise ConfigError(f"config '{config_name}.yaml' not found in {cfg_dir}")

    # version_base=None keeps Hydra in legacy/simple mode (no app version pin).
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        return compose(config_name=config_name, overrides=overrides or [])


def to_container(cfg: DictConfig) -> dict:
    """Resolve a DictConfig into a plain Python dict (interpolations expanded)."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


__all__ = ["CONFIGS_DIR", "load_config", "to_container"]
