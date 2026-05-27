"""Tests for the tool-execution sandbox (SPEC §6.7).

Covers policy enforcement (network / FS / timeout), result hashing, the
executor choke point and its full logging contract, and the orchestrator
wiring + production guard. Uses :class:`LocalSandbox` so no Docker is needed.
"""

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from common.types import ToolCall
from harness.sandbox import (
    DockerSandbox,
    LocalSandbox,
    PolicyViolation,
    ResourceLimitExceeded,
    SandboxedToolExecutor,
    SandboxError,
    SandboxPolicy,
    default_policy_for,
    enforce_policy,
    host_allowed,
    result_hash,
)

# --- test tools ---


class EchoTool:
    """Trivial offline tool: echoes its args back."""

    name = "echo"
    network_required = False

    async def __call__(self, args: dict, *, workdir: Path):
        return {"echoed": args}


class WriteTool:
    """Writes a file into its workdir (used to probe FS isolation)."""

    name = "write"
    network_required = False

    async def __call__(self, args: dict, *, workdir: Path):
        target = workdir / "out.txt"
        target.write_text(args.get("content", "hi"))
        return str(target)


class SlowTool:
    name = "slow"
    network_required = False

    async def __call__(self, args: dict, *, workdir: Path):
        await asyncio.sleep(5)
        return "done"


class FetchTool:
    """Declares it needs the network — exercises the network policy gate."""

    name = "fetch"
    network_required = True

    async def __call__(self, args: dict, *, workdir: Path):
        return "fetched"


class BoomTool:
    name = "boom"
    network_required = False

    async def __call__(self, args: dict, *, workdir: Path):
        raise RuntimeError("kaboom")


# --- SandboxPolicy validation ---


def test_policy_defaults_are_deny_by_default():
    p = SandboxPolicy()
    assert p.network == "none"
    assert p.cpu_seconds == 30
    assert p.memory_mb == 512
    assert p.fs_writable is True


def test_allowlist_requires_allowlist_mode():
    with pytest.raises(ValueError):
        SandboxPolicy(network="full", allowlist=["example.com"])


def test_allowlist_mode_requires_nonempty_list():
    with pytest.raises(ValueError):
        SandboxPolicy(network="allowlist", allowlist=[])


# --- result hashing ---


def test_result_hash_is_order_independent():
    assert result_hash({"a": 1, "b": 2}) == result_hash({"b": 2, "a": 1})


def test_result_hash_handles_non_json():
    # An un-serializable object falls back to repr without raising.
    assert isinstance(result_hash(object()), str)


# --- pre-flight policy enforcement ---


def test_enforce_policy_blocks_network_tool_under_none():
    with pytest.raises(PolicyViolation):
        enforce_policy(FetchTool(), SandboxPolicy(network="none"))


def test_enforce_policy_allows_network_tool_with_full():
    enforce_policy(FetchTool(), SandboxPolicy(network="full"))  # no raise


def test_host_allowed_modes():
    none = SandboxPolicy(network="none")
    full = SandboxPolicy(network="full")
    allow = SandboxPolicy(network="allowlist", allowlist=["example.com"])
    assert not host_allowed("example.com", none)
    assert host_allowed("anything.org", full)
    assert host_allowed("example.com", allow)
    assert host_allowed("api.example.com", allow)  # subdomain
    assert not host_allowed("evil.com", allow)


# --- LocalSandbox execution + isolation ---


async def test_local_sandbox_runs_tool_and_hashes_result():
    res = await LocalSandbox().run(EchoTool(), {"x": 1}, SandboxPolicy())
    assert res.ok
    assert res.output == {"echoed": {"x": 1}}
    assert res.result_hash == result_hash({"echoed": {"x": 1}})


async def test_local_sandbox_workdir_is_isolated_and_cleaned():
    # The tool writes into its workdir; that path must not survive the call.
    res = await LocalSandbox().run(WriteTool(), {"content": "secret"}, SandboxPolicy())
    assert res.ok
    assert not Path(res.output).exists()


async def test_local_sandbox_timeout_raises_resource_error():
    policy = SandboxPolicy(cpu_seconds=1)
    with pytest.raises(ResourceLimitExceeded):
        await LocalSandbox().run(SlowTool(), {}, policy)


async def test_local_sandbox_rejects_network_tool_before_running():
    with pytest.raises(PolicyViolation):
        await LocalSandbox().run(FetchTool(), {}, SandboxPolicy(network="none"))


# --- SandboxedToolExecutor: choke point + error capture ---


async def test_executor_returns_tool_result_on_success():
    ex = SandboxedToolExecutor(LocalSandbox())
    res = await ex.execute(EchoTool(), {"k": "v"})
    assert res.ok
    assert res.result_hash is not None


async def test_executor_captures_policy_violation_as_failed_result():
    ex = SandboxedToolExecutor(LocalSandbox())
    res = await ex.execute(FetchTool(), {}, policy=SandboxPolicy(network="none"))
    assert not res.ok
    assert "policy violation" in res.error


async def test_executor_captures_tool_exception():
    ex = SandboxedToolExecutor(LocalSandbox())
    res = await ex.execute(BoomTool(), {})
    assert not res.ok
    assert "tool error" in res.error
    assert "kaboom" in res.error


async def test_executor_captures_readonly_write_failure():
    ex = SandboxedToolExecutor(LocalSandbox())
    res = await ex.execute(WriteTool(), {}, policy=SandboxPolicy(fs_writable=False))
    assert not res.ok  # the write into a read-only workdir failed


async def test_executor_policy_resolution_order():
    floor = SandboxPolicy(cpu_seconds=10)
    per_tool = SandboxPolicy(cpu_seconds=20)
    override = SandboxPolicy(cpu_seconds=30)
    ex = SandboxedToolExecutor(LocalSandbox(), default_policy=floor, policies={"echo": per_tool})
    tool = EchoTool()
    assert ex.policy_for(tool) is per_tool  # per-tool beats floor
    assert ex.policy_for(tool, override) is override  # explicit beats all
    assert ex.policy_for(SlowTool()) is floor  # unknown tool falls to floor


def test_default_policy_for_is_no_network():
    assert default_policy_for(EchoTool()).network == "none"


# --- DockerSandbox guards (no daemon needed) ---


async def test_docker_sandbox_rejects_non_container_tool():
    # EchoTool is a callable Tool, not a ContainerTool — DockerSandbox refuses it.
    with pytest.raises(SandboxError):
        await DockerSandbox(client=object()).run(EchoTool(), {}, SandboxPolicy())


# --- orchestrator wiring (SPEC §6.7) ---


def _orch_imports():
    from harness.citation import CitationEnforcer, CitedDraft, CitedSegment
    from harness.context import DefaultPacker
    from harness.orchestrator import OrchestratorDeps, build_orchestrator, initial_state
    from harness.planning import ReactPlanner

    return (
        CitationEnforcer,
        CitedDraft,
        CitedSegment,
        DefaultPacker,
        OrchestratorDeps,
        build_orchestrator,
        initial_state,
        ReactPlanner,
    )


class _NullPipeline:
    cost = 0.0

    async def retrieve(self, query, k):
        from common.schemas import RetrievalResult

        return RetrievalResult(candidates=[], query=query, trace_id=uuid4(), cost=0.0)


def _deps_with_tools(**extra):
    (
        CitationEnforcer,
        CitedDraft,
        _CitedSegment,
        DefaultPacker,
        OrchestratorDeps,
        _build,
        _init,
        ReactPlanner,
    ) = _orch_imports()

    async def _draft(question, candidates):
        return CitedDraft(refused=True, refusal_reason="done")

    return OrchestratorDeps(
        pipeline=_NullPipeline(),
        enforcer=CitationEnforcer(draft_fn=_draft),
        packer=DefaultPacker(),
        planner=ReactPlanner(),
        **extra,
    )


async def test_orchestrator_runs_pending_tool_calls_through_sandbox():
    (
        *_,
        build_orchestrator,
        initial_state,
        _ReactPlanner,
    ) = _orch_imports()
    executor = SandboxedToolExecutor(LocalSandbox())
    deps = _deps_with_tools(tool_executor=executor, tools={"echo": EchoTool()})
    app = build_orchestrator(deps)
    state = initial_state("q", budget_usd=1.0, max_hops=0)
    state["pending_tool_calls"] = [ToolCall(tool="echo", args={"hi": 1})]
    cfg = {"configurable": {"thread_id": str(uuid4())}}
    final = await app.ainvoke(state, cfg)
    assert len(final["tool_results"]) == 1
    assert final["tool_results"][0].ok
    assert final["pending_tool_calls"] == []  # queue drained


async def test_orchestrator_unknown_tool_yields_failed_result():
    *_, build_orchestrator, initial_state, _ = _orch_imports()
    executor = SandboxedToolExecutor(LocalSandbox())
    deps = _deps_with_tools(tool_executor=executor, tools={})
    app = build_orchestrator(deps)
    state = initial_state("q", budget_usd=1.0, max_hops=0)
    state["pending_tool_calls"] = [ToolCall(tool="ghost", args={})]
    cfg = {"configurable": {"thread_id": str(uuid4())}}
    final = await app.ainvoke(state, cfg)
    assert not final["tool_results"][0].ok
    assert "unknown tool" in final["tool_results"][0].error


def test_production_guard_rejects_unsandboxed_tools():
    *_, build_orchestrator, _initial_state, _ = _orch_imports()
    deps = _deps_with_tools(tools={"echo": EchoTool()}, require_sandbox=True)
    with pytest.raises(ValueError, match="unsandboxed"):
        build_orchestrator(deps)
