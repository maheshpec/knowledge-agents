"""The single choke point for tool execution (SPEC §6.7, §6.9).

Every tool call in the system goes through :class:`SandboxedToolExecutor`. It:

1. resolves the policy for the call (explicit, per-tool default, or the
   executor's floor),
2. delegates execution to the wrapped :class:`~harness.sandbox.base.Sandbox`,
3. logs the call — tool, sandbox policy, args, and result hash — per the
   observability contract (SPEC §6.9), and
4. converts any sandbox/tool failure into a :class:`ToolResult` so a single bad
   tool call never crashes the orchestrator graph.

Routing all execution through one object is what lets the orchestrator promise
that *no tool runs unsandboxed in prod* (SPEC §13 anti-pattern): the graph holds
an executor, never a raw tool.
"""

from __future__ import annotations

from typing import Any

from common.types import ToolResult
from harness.observability.logging import get_logger
from harness.observability.tracing import traced
from harness.sandbox.base import (
    PolicyViolation,
    ResourceLimitExceeded,
    Sandbox,
    SandboxError,
    SandboxPolicy,
    Tool,
    default_policy_for,
)

_log = get_logger("harness.sandbox.executor")


class SandboxedToolExecutor:
    """Wrap a :class:`Sandbox` with policy resolution + full call logging."""

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        default_policy: SandboxPolicy | None = None,
        policies: dict[str, SandboxPolicy] | None = None,
    ) -> None:
        self._sandbox = sandbox
        # Executor-wide floor used when a tool has no specific policy.
        self._default = default_policy
        # Optional per-tool overrides keyed by tool name.
        self._policies = policies or {}

    def policy_for(self, tool: Tool, override: SandboxPolicy | None = None) -> SandboxPolicy:
        """Resolve the policy: explicit override > per-tool > floor > tool default."""
        if override is not None:
            return override
        if tool.name in self._policies:
            return self._policies[tool.name]
        if self._default is not None:
            return self._default
        return default_policy_for(tool)

    @traced(span_name="sandbox.execute")
    async def execute(
        self, tool: Tool, args: dict[str, Any], *, policy: SandboxPolicy | None = None
    ) -> ToolResult:
        resolved = self.policy_for(tool, policy)
        # The §6.9 record: every tool call logs its policy, args, and result hash.
        # Logged before execution so even a crash leaves the policy+args on record.
        _log.info(
            "sandbox.tool_call",
            tool=tool.name,
            policy=resolved.model_dump(),
            args=args,
        )
        try:
            result = await self._sandbox.run(tool, args, resolved)
        except PolicyViolation as exc:
            _log.warning("sandbox.policy_violation", tool=tool.name, error=str(exc))
            return ToolResult(tool=tool.name, ok=False, error=f"policy violation: {exc}")
        except ResourceLimitExceeded as exc:
            _log.warning("sandbox.resource_exceeded", tool=tool.name, error=str(exc))
            return ToolResult(tool=tool.name, ok=False, error=f"resource limit: {exc}")
        except SandboxError as exc:
            _log.error("sandbox.error", tool=tool.name, error=str(exc))
            return ToolResult(tool=tool.name, ok=False, error=f"sandbox error: {exc}")
        except Exception as exc:  # noqa: BLE001 - a tool fault must not crash the graph
            _log.error("sandbox.tool_error", tool=tool.name, error=str(exc))
            return ToolResult(tool=tool.name, ok=False, error=f"tool error: {exc}")

        _log.info(
            "sandbox.tool_result",
            tool=tool.name,
            ok=result.ok,
            result_hash=result.result_hash,
        )
        return result


__all__ = ["SandboxedToolExecutor"]
