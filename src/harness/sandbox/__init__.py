"""Tool-execution sandbox (SPEC §6.7).

Isolated, policy-enforced execution for tool calls, with full per-call logging
(SPEC §6.9). The public surface:

* :class:`SandboxPolicy` — network / CPU / memory / FS limits for one call.
* :class:`Tool` / :class:`Sandbox` — the execution contracts.
* :class:`LocalSandbox` — in-process backend for trusted tools and tests.
* :class:`DockerSandbox` — ephemeral-container backend for untrusted execution.
* :class:`SandboxedToolExecutor` — the choke point that policies + logs every
  call; the orchestrator holds this, never a raw tool.
"""

from __future__ import annotations

from harness.sandbox.base import (
    NetworkMode,
    PolicyViolation,
    ResourceLimitExceeded,
    Sandbox,
    SandboxError,
    SandboxPolicy,
    Tool,
    default_policy_for,
    result_hash,
)
from harness.sandbox.docker import ContainerTool, DockerSandbox
from harness.sandbox.executor import SandboxedToolExecutor
from harness.sandbox.local import LocalSandbox
from harness.sandbox.policy import enforce_policy, host_allowed

__all__ = [
    "NetworkMode",
    "SandboxPolicy",
    "Tool",
    "Sandbox",
    "SandboxError",
    "PolicyViolation",
    "ResourceLimitExceeded",
    "result_hash",
    "default_policy_for",
    "enforce_policy",
    "host_allowed",
    "LocalSandbox",
    "DockerSandbox",
    "ContainerTool",
    "SandboxedToolExecutor",
]
