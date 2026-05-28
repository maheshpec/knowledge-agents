"""DCI sandbox policy (SPEC §15.4, §6.7).

The DCI tools run inside the Phase 3N sandbox under a deny-by-default policy:

* **No network.** A grep / glob / read against the corpus never leaves the
  sandbox; the index itself is mounted (or, in the in-memory dev store, lives
  in the same process). This is the line of defence against a prompt-injection
  payload that asked the agent to ``corpus_read`` then exfiltrate the result.
* **Read-only workdir.** Tools must not write back into the corpus or onto
  disk; they return values through the executor.
* **Bounded CPU + memory.** A pathological regex or a very large ``max_bytes``
  read is killed by the backend before it can burn the budget.

The function returned here builds the policy fresh per call so tuning knobs
(``cpu_seconds``, ``memory_mb``) can be overridden without mutating shared
state.
"""

from __future__ import annotations

from harness.sandbox import SandboxPolicy

# Defaults are conservative: enough headroom for a 50k-byte read or a 50-hit
# regex sweep over a ~1k-doc corpus, but tight enough that a runaway tool gets
# killed quickly.
DEFAULT_CPU_SECONDS = 15
DEFAULT_MEMORY_MB = 512


def dci_policy(
    *,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> SandboxPolicy:
    """Build the default sandbox policy for DCI tools (SPEC §15.4)."""
    return SandboxPolicy(
        network="none",
        cpu_seconds=cpu_seconds,
        memory_mb=memory_mb,
        fs_writable=False,
    )


__all__ = ["dci_policy", "DEFAULT_CPU_SECONDS", "DEFAULT_MEMORY_MB"]
