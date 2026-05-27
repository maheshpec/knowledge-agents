"""GitHub + git-branch contracts for PR creation (SPEC §8.4).

Two injectable surfaces keep the generator testable and side-effect-free under
test:

* :class:`GitHubClient` — opens a pull request. **It deliberately exposes no
  merge operation.** The "no auto-merge" rule (SPEC §8.4 / anti-pattern §13) is
  enforced *structurally*: there is no API on this protocol the generator could
  call to merge, so it cannot, regardless of logic bugs. A human merges via the
  GitHub UI.
* :class:`BranchWriter` — creates the branch, writes the config file, and commits.

The real implementations (:class:`GitHubRESTClient`, :class:`GitBranchWriter`)
import their backends lazily; tests use the in-memory doubles
(:class:`RecordingGitHubClient`, :class:`RecordingBranchWriter`).
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from common.errors import KnowledgeAgentError


class PRGenerationError(KnowledgeAgentError):
    """Raised when a PR cannot be generated (e.g. non-accepted candidate, git failure)."""


class PROpenRequest(BaseModel):
    """A request to open a pull request. ``draft=True`` by default (SPEC §8.4)."""

    title: str
    body: str
    head_branch: str
    base_branch: str = "main"
    draft: bool = True  # opened as draft so it is never mistaken for merge-ready
    labels: list[str] = Field(default_factory=list)


class OpenedPR(BaseModel):
    """The result of opening a pull request."""

    number: int
    url: str
    head_branch: str
    draft: bool = True


@runtime_checkable
class GitHubClient(Protocol):
    """Open a pull request. No merge method exists — by design (SPEC §8.4 / §13)."""

    async def open_pull_request(self, req: PROpenRequest) -> OpenedPR: ...


@runtime_checkable
class BranchWriter(Protocol):
    """Create a branch, write files into it, and commit (SPEC §8.4)."""

    async def create_branch(self, name: str, base: str) -> None: ...

    async def write_file(self, path: str, content: str) -> None: ...

    async def commit(self, message: str) -> None: ...


# --- in-memory test doubles ------------------------------------------------


class RecordingGitHubClient:
    """Test double: records opened PRs, fabricates a deterministic number/URL.

    Has no merge method (mirrors the protocol), so a test can assert the merge
    surface simply does not exist. ``opened`` exposes every PR request seen.
    """

    def __init__(self, *, repo: str = "owner/repo", start_number: int = 100) -> None:
        self._repo = repo
        self._next = start_number
        self.opened: list[PROpenRequest] = []

    async def open_pull_request(self, req: PROpenRequest) -> OpenedPR:
        self.opened.append(req)
        number = self._next
        self._next += 1
        return OpenedPR(
            number=number,
            url=f"https://github.com/{self._repo}/pull/{number}",
            head_branch=req.head_branch,
            draft=req.draft,
        )


class RecordingBranchWriter:
    """Test double: records branch ops + written files in memory (no git)."""

    def __init__(self) -> None:
        self.branch: tuple[str, str] | None = None  # (name, base)
        self.files: dict[str, str] = {}
        self.commits: list[str] = []

    async def create_branch(self, name: str, base: str) -> None:
        self.branch = (name, base)

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    async def commit(self, message: str) -> None:
        self.commits.append(message)


# --- real implementations (lazy backends) ----------------------------------


class GitBranchWriter:
    """Branch/write/commit against a real git working tree via ``git`` subprocess.

    Side-effecting; used in production only. Each call shells out to ``git`` in
    ``repo_root``; failures raise :class:`PRGenerationError` with stderr attached.
    """

    def __init__(self, repo_root: str = ".") -> None:
        self._root = repo_root

    async def _git(self, *args: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=self._root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise PRGenerationError(f"git {' '.join(args)} failed: {stderr.decode().strip()}")

    async def create_branch(self, name: str, base: str) -> None:
        await self._git("checkout", "-b", name, base)

    async def write_file(self, path: str, content: str) -> None:
        import os

        full = os.path.join(self._root, path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        # Off-loop blocking write is fine here (small config file, prod-only path).
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)
        await self._git("add", path)

    async def commit(self, message: str) -> None:
        await self._git("commit", "-m", message)


class GitHubRESTClient:
    """Open PRs via the GitHub REST API (``POST /repos/{repo}/pulls``).

    Lazily imports ``httpx``. Token + repo are injected; the client never exposes
    a merge call. Prod-only; tests use :class:`RecordingGitHubClient`.
    """

    def __init__(self, repo: str, token: str, *, base_url: str = "https://api.github.com") -> None:
        self._repo = repo
        self._token = token
        self._base_url = base_url.rstrip("/")

    async def open_pull_request(self, req: PROpenRequest) -> OpenedPR:
        import httpx  # lazy: avoids importing the HTTP stack in unit tests

        url = f"{self._base_url}/repos/{self._repo}/pulls"
        payload = {
            "title": req.title,
            "body": req.body,
            "head": req.head_branch,
            "base": req.base_branch,
            "draft": req.draft,
        }
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 300:
            raise PRGenerationError(f"GitHub PR creation failed [{resp.status_code}]: {resp.text}")
        data = resp.json()
        opened = OpenedPR(
            number=data["number"],
            url=data["html_url"],
            head_branch=req.head_branch,
            draft=req.draft,
        )
        if req.labels:
            await self._add_labels(client_repo=self._repo, number=opened.number, labels=req.labels)
        return opened

    async def _add_labels(self, *, client_repo: str, number: int, labels: list[str]) -> None:
        import httpx

        url = f"{self._base_url}/repos/{client_repo}/issues/{number}/labels"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
        }
        async with httpx.AsyncClient() as client:
            await client.post(url, json={"labels": labels}, headers=headers)


__all__ = [
    "PRGenerationError",
    "PROpenRequest",
    "OpenedPR",
    "GitHubClient",
    "BranchWriter",
    "RecordingGitHubClient",
    "RecordingBranchWriter",
    "GitBranchWriter",
    "GitHubRESTClient",
]
