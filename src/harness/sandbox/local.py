"""In-process sandbox for trusted Python tools (SPEC §6.7).

``LocalSandbox`` runs a :class:`~harness.sandbox.base.Tool` callable in this
process, under three controls:

* **Time** — the call is wrapped in ``asyncio.wait_for``; overrun raises
  :class:`ResourceLimitExceeded` (mapped to the policy's ``cpu_seconds`` as a
  wall-clock ceiling).
* **File system** — each call gets a fresh job-scoped temp directory as its
  ``workdir``; when ``fs_writable=False`` the directory is made read-only so a
  tool cannot persist anything.
* **Network** — only the *declared* need is enforced (via :func:`enforce_policy`);
  in-process execution cannot truly block sockets.

Because it shares the interpreter, ``LocalSandbox`` is **not** an isolation
boundary against hostile code — use :class:`~harness.sandbox.docker.DockerSandbox`
for untrusted execution. It exists for fast, dependency-free execution of
first-party tools and for testing the policy/logging machinery.
"""

from __future__ import annotations

import asyncio
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

from common.types import ToolResult
from harness.observability.logging import get_logger
from harness.sandbox.base import (
    ResourceLimitExceeded,
    SandboxPolicy,
    Tool,
    result_hash,
)
from harness.sandbox.policy import enforce_policy

_log = get_logger("harness.sandbox.local")


class LocalSandbox:
    """Run a tool in-process with a timeout and a job-scoped working directory."""

    def __init__(self, *, root: Path | None = None) -> None:
        # Parent dir for per-call temp workdirs; defaults to the OS temp location.
        self._root = root

    async def run(self, tool: Tool, args: dict[str, Any], policy: SandboxPolicy) -> ToolResult:
        # Pre-flight: reject before allocating anything.
        enforce_policy(tool, policy)

        workdir = Path(tempfile.mkdtemp(prefix="ka-sandbox-", dir=self._root))
        try:
            if not policy.fs_writable:
                _make_readonly(workdir)
            output = await asyncio.wait_for(tool(args, workdir=workdir), timeout=policy.cpu_seconds)
            return ToolResult(
                tool=tool.name, ok=True, output=output, result_hash=result_hash(output)
            )
        except TimeoutError as exc:
            # asyncio.wait_for raises TimeoutError on overrun. Surface it as a
            # resource breach so callers can distinguish it from a tool error.
            raise ResourceLimitExceeded(
                f"tool {tool.name!r} exceeded cpu_seconds={policy.cpu_seconds}"
            ) from exc
        finally:
            # Restore writability so cleanup can remove a read-only tree.
            if not policy.fs_writable:
                _make_writable(workdir)
            shutil.rmtree(workdir, ignore_errors=True)


def _make_readonly(path: Path) -> None:
    """Strip write bits from ``path`` and everything under it."""
    for p in [path, *path.rglob("*")]:
        p.chmod(p.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def _make_writable(path: Path) -> None:
    for p in [path, *path.rglob("*")]:
        try:
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
        except FileNotFoundError:  # pragma: no cover - race with tool cleanup
            pass


__all__ = ["LocalSandbox"]
