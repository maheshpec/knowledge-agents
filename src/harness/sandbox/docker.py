"""Ephemeral-container sandbox for untrusted execution (SPEC §6.7).

``DockerSandbox`` runs each tool call in a throwaway container with real
isolation: no network by default, a single bind-mounted job-scoped temp dir, and
hard CPU / memory / wall-clock ceilings. This is the production backend for
anything that could be hijacked by injected content (``code_execution``,
``web_fetch``).

The ``docker`` SDK is imported lazily so the core install and the unit-test path
(which use :class:`~harness.sandbox.local.LocalSandbox`) need neither the SDK nor
a running daemon. Install it with the ``sandbox`` extra.

Tools run here implement :class:`ContainerTool`: they name an ``image`` and
render a ``command`` from their args, rather than being Python callables (a
callable cannot cross the container boundary).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from common.types import ToolResult
from harness.observability.logging import get_logger
from harness.sandbox.base import (
    ResourceLimitExceeded,
    SandboxError,
    SandboxPolicy,
    ToolMeta,
    result_hash,
)
from harness.sandbox.policy import enforce_policy

_log = get_logger("harness.sandbox.docker")

# Where the job-scoped temp dir is mounted inside the container.
WORKSPACE = "/workspace"


@runtime_checkable
class ContainerTool(ToolMeta, Protocol):
    """A tool runnable as a container command (the shape ``DockerSandbox`` needs)."""

    image: str

    def command(self, args: dict[str, Any]) -> list[str]:
        """Render the container command (argv) for ``args``."""
        ...


class DockerSandbox:
    """Run each tool call in an ephemeral, resource-capped Docker container."""

    def __init__(self, *, client: Any = None) -> None:
        # Allow injecting a client (tests / custom config); otherwise built lazily.
        self._client = client

    def _docker_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import docker
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise SandboxError(
                "DockerSandbox needs the 'docker' SDK; install the 'sandbox' extra"
            ) from exc
        self._client = docker.from_env()
        return self._client

    async def run(self, tool: Any, args: dict[str, Any], policy: SandboxPolicy) -> ToolResult:
        if not isinstance(tool, ContainerTool):
            raise SandboxError(
                f"DockerSandbox requires a ContainerTool (image+command); got {type(tool).__name__}"
            )
        enforce_policy(tool, policy)

        import asyncio
        import tempfile

        workdir = Path(tempfile.mkdtemp(prefix="ka-docker-"))
        # Offload the blocking docker calls to a thread so we don't stall the loop.
        try:
            return await asyncio.to_thread(self._run_blocking, tool, args, policy, workdir)
        finally:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)

    def _run_blocking(
        self, tool: ContainerTool, args: dict[str, Any], policy: SandboxPolicy, workdir: Path
    ) -> ToolResult:
        client = self._docker_client()
        host_config = self._container_kwargs(policy, workdir)
        _log.info(
            "sandbox.docker.start",
            tool=tool.name,
            image=tool.image,
            network=policy.network,
            cpu_seconds=policy.cpu_seconds,
            memory_mb=policy.memory_mb,
        )
        container = client.containers.run(
            tool.image,
            command=tool.command(args),
            detach=True,
            **host_config,
        )
        try:
            try:
                status = container.wait(timeout=policy.cpu_seconds)
            except Exception as exc:  # docker raises on read timeout
                container.kill()
                raise ResourceLimitExceeded(
                    f"tool {tool.name!r} exceeded cpu_seconds={policy.cpu_seconds}"
                ) from exc
            logs = container.logs().decode("utf-8", errors="replace")
            exit_code = status.get("StatusCode", 0) if isinstance(status, dict) else status
            if exit_code != 0:
                return ToolResult(
                    tool=tool.name, ok=False, error=f"exit {exit_code}: {logs[-2000:]}"
                )
            return ToolResult(tool=tool.name, ok=True, output=logs, result_hash=result_hash(logs))
        finally:
            container.remove(force=True)

    def _container_kwargs(self, policy: SandboxPolicy, workdir: Path) -> dict[str, Any]:
        """Translate a :class:`SandboxPolicy` into ``docker run`` arguments."""
        kwargs: dict[str, Any] = {
            # nano_cpus caps CPU; mem_limit caps memory; both are hard limits.
            "mem_limit": f"{policy.memory_mb}m",
            "nano_cpus": 1_000_000_000,  # 1 CPU; wall-clock is bounded by wait(timeout)
            "read_only": not policy.fs_writable,
            "volumes": {str(workdir): {"bind": WORKSPACE, "mode": "rw"}},
            "working_dir": WORKSPACE,
            # Drop privileges + capabilities; no host namespaces.
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges"],
            "pids_limit": 128,
        }
        # network="none" disables networking entirely; allowlist/full keep the
        # default bridge (host-level filtering applies the allowlist).
        if policy.network == "none":
            kwargs["network_disabled"] = True
        return kwargs


__all__ = ["DockerSandbox", "ContainerTool", "WORKSPACE"]
