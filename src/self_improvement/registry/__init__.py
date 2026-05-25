"""Component registry (SPEC §8.1): the bounded, declared search space."""

from self_improvement.registry.pipeline_config import (
    PipelineConfig,
    pipeline_from_registry,
)
from self_improvement.registry.registry import (
    CATEGORIES,
    ComponentRegistry,
    RegistryDeps,
)
from self_improvement.registry.spec import (
    ComponentSpec,
    ParamSpec,
    RegistryError,
)

__all__ = [
    "CATEGORIES",
    "ComponentRegistry",
    "RegistryDeps",
    "ComponentSpec",
    "ParamSpec",
    "RegistryError",
    "PipelineConfig",
    "pipeline_from_registry",
]
