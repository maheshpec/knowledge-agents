"""Component + parameter specs parsed from ``configs/components.yaml`` (SPEC §8.1).

These dataclasses are the in-memory shape of the registry's declared search
space. :class:`ParamSpec` knows how to validate and sample a single parameter;
:class:`ComponentSpec` bundles a component's identity with its tunable params.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal

from common.errors import KnowledgeAgentError

ParamType = Literal["int", "float", "enum"]


class RegistryError(KnowledgeAgentError):
    """Raised on malformed registry specs or invalid parameter values (SPEC §8.1)."""


@dataclass(frozen=True)
class ParamSpec:
    """A single tunable parameter: its type, bounds/choices, and default."""

    name: str
    type: ParamType
    default: Any
    range: tuple[float, float] | None = None  # numeric (int/float)
    values: list[Any] | None = None  # enum choices

    @classmethod
    def from_yaml(cls, name: str, raw: dict[str, Any]) -> ParamSpec:
        ptype: ParamType = raw.get("type", "enum")
        rng = raw.get("range")
        return cls(
            name=name,
            type=ptype,
            default=raw.get("default"),
            range=(float(rng[0]), float(rng[1])) if rng else None,
            values=list(raw["values"]) if "values" in raw else None,
        )

    def validate(self, value: Any) -> Any:
        """Coerce + bounds/choice-check a value; raise :class:`RegistryError` if invalid."""
        if self.type == "int":
            ivalue = int(value)
            self._check_range(ivalue)
            return ivalue
        if self.type == "float":
            fvalue = float(value)
            self._check_range(fvalue)
            return fvalue
        # enum
        if self.values is not None and value not in self.values:
            raise RegistryError(
                f"param '{self.name}'={value!r} not in allowed values {self.values}"
            )
        return value

    def _check_range(self, value: float) -> None:
        if self.range is not None and not (self.range[0] <= value <= self.range[1]):
            raise RegistryError(
                f"param '{self.name}'={value} out of range [{self.range[0]}, {self.range[1]}]"
            )

    def sample(self, rng: random.Random) -> Any:
        """Draw a value within the declared range / from the declared choices."""
        if self.type == "int" and self.range is not None:
            return rng.randint(int(self.range[0]), int(self.range[1]))
        if self.type == "float" and self.range is not None:
            return rng.uniform(self.range[0], self.range[1])
        if self.values:
            return rng.choice(self.values)
        return self.default


@dataclass(frozen=True)
class ComponentSpec:
    """A registered component: category + name, optional class path, tunable params."""

    category: str
    name: str
    class_path: str | None = None
    params: dict[str, ParamSpec] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, category: str, raw: dict[str, Any]) -> ComponentSpec:
        params = {
            pname: ParamSpec.from_yaml(pname, pdef)
            for pname, pdef in (raw.get("params") or {}).items()
        }
        return cls(
            category=category,
            name=str(raw["name"]),
            class_path=raw.get("class"),
            params=params,
        )

    def defaults(self) -> dict[str, Any]:
        return {name: spec.default for name, spec in self.params.items()}


__all__ = ["ParamType", "RegistryError", "ParamSpec", "ComponentSpec"]
