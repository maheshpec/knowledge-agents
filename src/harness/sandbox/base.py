"""Sandbox contracts (SPEC §6.7).

Tool-execution isolation. Anything a tool touches — the network, the file
system, CPU/memory — is constrained by a :class:`SandboxPolicy`. The policy is
*deny by default*: no network, a job-scoped temp dir, and hard CPU/memory/time
ceilings. This is the line of defence against a prompt-injection payload that
rode in on retrieved content and tries to make ``code_execution`` or
``web_fetch`` reach out or exfiltrate.

Two backends implement the :class:`Sandbox` protocol:

* :class:`~harness.sandbox.local.LocalSandbox` — runs a Python ``Tool`` callable
  in-process with a timeout and a fresh working directory. Cheap, no Docker,
  good for trusted first-party tools and for unit tests.
* :class:`~harness.sandbox.docker.DockerSandbox` — runs each call in an
  ephemeral container with real network/FS/resource isolation. The production
  choice for untrusted execution.

Both run *through* :class:`~harness.sandbox.executor.SandboxedToolExecutor`,
which logs every call (policy, args, result hash — SPEC §6.9) and is the single
choke point guaranteeing no tool runs unsandboxed in prod.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from common.types import ToolResult

NetworkMode = Literal["none", "allowlist", "full"]


class SandboxPolicy(BaseModel):
    """Resource + network policy enforced for a single tool call (SPEC §6.7).

    Defaults are the safe floor: no network, modest CPU/memory ceilings, and a
    writable job-scoped temp dir (the only path a tool may write to).
    """

    network: NetworkMode = "none"
    allowlist: list[str] = Field(default_factory=list)
    cpu_seconds: int = 30
    memory_mb: int = 512
    fs_writable: bool = True

    def model_post_init(self, _ctx: Any) -> None:  # noqa: D401
        # An allowlist without ``network="allowlist"`` is almost always a mistake;
        # and ``allowlist`` mode with an empty list denies everything, which is a
        # silent footgun. Surface both at construction rather than at run time.
        if self.network == "allowlist" and not self.allowlist:
            raise ValueError("network='allowlist' requires a non-empty allowlist")
        if self.allowlist and self.network != "allowlist":
            raise ValueError(
                f"allowlist is only meaningful with network='allowlist', got network={self.network!r}"
            )


@runtime_checkable
class ToolMeta(Protocol):
    """The tool metadata the policy layer needs, independent of how it runs.

    Both the in-process :class:`Tool` (an async callable) and the container
    :class:`~harness.sandbox.docker.ContainerTool` (an image + command) satisfy
    this, so :func:`~harness.sandbox.policy.enforce_policy` can pre-flight either
    without caring about the execution shape.
    """

    name: str
    network_required: bool


@runtime_checkable
class Tool(ToolMeta, Protocol):
    """A unit of executable work the orchestrator hands to a sandbox.

    A tool is an async callable plus metadata. ``network_required`` lets the
    policy layer reject a tool *before* it runs when the policy forbids the
    network it needs, rather than discovering the violation mid-execution.
    """

    async def __call__(self, args: dict[str, Any], *, workdir: Path) -> Any:
        """Run the tool against ``args``, writing only under ``workdir``."""
        ...


@runtime_checkable
class Sandbox(Protocol):
    """Executes a tool under a policy and returns a :class:`ToolResult` (SPEC §6.7)."""

    async def run(self, tool: Tool, args: dict[str, Any], policy: SandboxPolicy) -> ToolResult: ...


class SandboxError(Exception):
    """Base class for sandbox failures."""


class PolicyViolation(SandboxError):
    """A tool call was rejected because it violated its :class:`SandboxPolicy`."""


class ResourceLimitExceeded(SandboxError):
    """A tool call exceeded a CPU / memory / wall-clock limit and was killed."""


def result_hash(output: Any) -> str:
    """Stable SHA-256 of a tool's output for the observability record (SPEC §6.9).

    Hashes a canonical JSON encoding when possible (sorted keys, no whitespace
    drift) and falls back to the ``repr`` for non-JSON-serializable outputs, so
    the same logical result always hashes the same regardless of dict ordering.
    """
    try:
        encoded = json.dumps(output, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        encoded = repr(output)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def default_policy_for(tool: Tool) -> SandboxPolicy:
    """The safe default policy for ``tool``.

    Network-requiring tools (e.g. ``web_fetch``) get ``allowlist`` mode but with
    an *empty intent* — callers must supply the hosts. Since an empty allowlist
    is rejected, the default for a network tool is ``full`` only when explicitly
    chosen; here we keep the floor at ``none`` and let the orchestrator widen it.
    """
    return SandboxPolicy(network="none")


__all__ = [
    "NetworkMode",
    "SandboxPolicy",
    "ToolMeta",
    "Tool",
    "Sandbox",
    "SandboxError",
    "PolicyViolation",
    "ResourceLimitExceeded",
    "result_hash",
    "default_policy_for",
]
