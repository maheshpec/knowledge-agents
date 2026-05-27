"""Policy enforcement, evaluated *before* a tool runs (SPEC §6.7).

Pre-flight checks are pure and cheap, so they live apart from the execution
backends: both :class:`~harness.sandbox.local.LocalSandbox` and
:class:`~harness.sandbox.docker.DockerSandbox` call :func:`enforce_policy` first
and bail out with a :class:`PolicyViolation` before spending any compute. The
hard resource ceilings (CPU/memory/timeout) are enforced *during* execution by
the backend, since only the backend can kill a runaway call.
"""

from __future__ import annotations

from harness.sandbox.base import PolicyViolation, SandboxPolicy, ToolMeta


def enforce_policy(tool: ToolMeta, policy: SandboxPolicy) -> None:
    """Reject a tool whose declared needs the policy forbids. Raises on violation.

    The one statically checkable contract is the network: a tool that declares
    ``network_required`` cannot run under ``network="none"``. Everything else
    (CPU, memory, wall-clock) can only be observed while the tool runs, so it is
    enforced by the backend, not here.
    """
    if tool.network_required and policy.network == "none":
        raise PolicyViolation(
            f"tool {tool.name!r} requires network access but policy is network='none'"
        )


def host_allowed(host: str, policy: SandboxPolicy) -> bool:
    """Whether ``host`` may be reached under ``policy``.

    ``full`` allows everything, ``none`` allows nothing, ``allowlist`` allows an
    exact host match or a subdomain of an allowlisted entry (``api.example.com``
    is allowed by ``example.com``). The orchestrator's network-aware tools use
    this to gate individual outbound requests.
    """
    if policy.network == "full":
        return True
    if policy.network == "none":
        return False
    host = host.lower().strip(".")
    for entry in policy.allowlist:
        entry = entry.lower().strip(".")
        if host == entry or host.endswith(f".{entry}"):
            return True
    return False


__all__ = ["enforce_policy", "host_allowed"]
