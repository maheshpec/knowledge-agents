"""DCI sandbox isolation tests (SPEC §15.4).

The Phase 3N sandbox is the line of defence between the DCI tools and the
host: no network, no writes, time/memory caps. These tests run each tool
through :class:`LocalSandbox` + :class:`SandboxedToolExecutor` and verify:

* the default DCI policy is deny-by-default (no network, FS read-only);
* a write attempt under the policy fails inside the sandbox;
* tools that did NOT declare ``network_required`` still get rejected if the
  policy is later widened to require it (defensive);
* the executor turns a tool exception into a failed ``ToolResult`` instead of
  crashing the graph (matches the §6.7 contract).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from common.schemas import Chunk
from harness.sandbox import (
    LocalSandbox,
    PolicyViolation,
    SandboxedToolExecutor,
    SandboxPolicy,
    enforce_policy,
)
from knowledge_index.dci import (
    CorpusGrepTool,
    CorpusReadTool,
    InMemoryCorpusStore,
    dci_policy,
    make_dci_tools,
)
from knowledge_index.graph.store import InMemoryGraphStore


def _chunk(doc_id, chunk_id, text):
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        metadata={"collection": "main", "source": "docs", "type": "md"},
    )


@pytest.fixture
def store() -> InMemoryCorpusStore:
    s = InMemoryCorpusStore()
    s.add_chunks([_chunk("doc-a", "doc-a:0", "line one\nfoo line two\nline three")])
    return s


# --- policy defaults ----


def test_dci_policy_denies_network_and_writes():
    p = dci_policy()
    assert p.network == "none"
    assert p.fs_writable is False
    assert p.cpu_seconds > 0
    assert p.memory_mb > 0


def test_dci_policy_tunable():
    p = dci_policy(cpu_seconds=5, memory_mb=128)
    assert p.cpu_seconds == 5
    assert p.memory_mb == 128


# --- enforce_policy: tools declare no network, policy denies it -> OK ----


def test_all_dci_tools_pass_default_policy_preflight(store: InMemoryCorpusStore):
    tools = make_dci_tools(store, InMemoryGraphStore())
    policy = dci_policy()
    for tool in tools.values():
        enforce_policy(tool, policy)  # no raise


# --- running through the executor ----


async def test_grep_tool_runs_under_sandbox_and_returns_ok(
    store: InMemoryCorpusStore,
):
    ex = SandboxedToolExecutor(LocalSandbox(), default_policy=dci_policy())
    res = await ex.execute(CorpusGrepTool(store), {"pattern": "foo"})
    assert res.ok
    # GrepHit list serializes; the executor records a hash for §6.9.
    assert res.result_hash is not None
    assert res.output[0].snippet.startswith("foo")


async def test_read_tool_runs_under_readonly_workdir(store: InMemoryCorpusStore):
    # ``fs_writable=False`` makes the workdir read-only; the read tool never
    # writes to disk, so it must still succeed.
    ex = SandboxedToolExecutor(LocalSandbox(), default_policy=dci_policy())
    res = await ex.execute(CorpusReadTool(store), {"doc_id": "doc-a"})
    assert res.ok
    assert "line one" in res.output.content


async def test_write_attempt_under_dci_policy_fails():
    """A DCI tool that tried to persist files would be killed by the policy."""

    class _WriteAttempt:
        """Simulates a hypothetical leaky tool that tries to write to disk."""

        name = "leaky"
        network_required = False

        async def __call__(self, args: dict[str, Any], *, workdir: Path):
            (workdir / "leak.txt").write_text("exfil")
            return "leaked"

    ex = SandboxedToolExecutor(LocalSandbox(), default_policy=dci_policy())
    res = await ex.execute(_WriteAttempt(), {})
    # Under ``fs_writable=False`` the workdir is read-only, so the write
    # raises and the executor surfaces a failed ToolResult.
    assert not res.ok
    assert "tool error" in (res.error or "")


# --- defensive: if a DCI tool were tagged network_required, policy blocks ----


def test_policy_blocks_tool_that_claims_network():
    class _Pretender:
        name = "pretender"
        network_required = True

        async def __call__(self, args, *, workdir):
            return None

    with pytest.raises(PolicyViolation):
        enforce_policy(_Pretender(), dci_policy())


# --- a tool exception is captured, not raised ----


async def test_tool_exception_becomes_failed_result(store: InMemoryCorpusStore):
    ex = SandboxedToolExecutor(LocalSandbox(), default_policy=dci_policy())
    # Missing ``pattern`` is a KnowledgeAgentError inside the tool; the executor
    # must convert it to a failed ToolResult per SPEC §6.7.
    res = await ex.execute(CorpusGrepTool(store), {})
    assert not res.ok
    assert "tool error" in (res.error or "")


# --- escape attempts: regex backreference / arbitrary input ----


async def test_grep_does_not_escape_via_malformed_regex(store: InMemoryCorpusStore):
    ex = SandboxedToolExecutor(LocalSandbox(), default_policy=dci_policy())
    # Pathological regex must not crash; the tool falls back to literal match.
    res = await ex.execute(CorpusGrepTool(store), {"pattern": "[unterminated"})
    assert res.ok
    assert res.output == []


async def test_sandboxed_default_policy_is_no_network():
    """Sanity: the default policy a sandbox falls back to is still no-net."""
    from harness.sandbox import default_policy_for

    tool = CorpusGrepTool(InMemoryCorpusStore())
    assert default_policy_for(tool).network == "none"


# --- principals carried through the sandbox boundary ----


async def test_principals_pass_through_sandbox(store: InMemoryCorpusStore):
    # Add a private doc; principal-less call must not see it via the sandbox.
    store.add_chunks([_chunk("doc-priv", "doc-priv:0", "private secret")])
    # Replace with an ACL'd version.
    store2 = InMemoryCorpusStore()
    store2.add_chunks(
        [
            Chunk(
                chunk_id="doc-priv:0",
                doc_id="doc-priv",
                text="private secret",
                acl=["team-x"],
                metadata={"collection": "main", "source": "docs", "type": "md"},
            )
        ]
    )
    ex = SandboxedToolExecutor(LocalSandbox(), default_policy=dci_policy())
    res = await ex.execute(CorpusGrepTool(store2), {"pattern": "secret"})
    assert res.ok and res.output == []
    res = await ex.execute(
        CorpusGrepTool(store2), {"pattern": "secret", "user_principals": ["team-x"]}
    )
    assert res.ok and len(res.output) == 1


# --- explicit policy override is honored ----


async def test_explicit_policy_override_honored(store: InMemoryCorpusStore):
    ex = SandboxedToolExecutor(LocalSandbox())
    res = await ex.execute(
        CorpusGrepTool(store),
        {"pattern": "line"},
        policy=SandboxPolicy(network="none", fs_writable=False, cpu_seconds=5),
    )
    assert res.ok
