"""Typed exceptions raised across the harness (SPEC §6).

Catching these by type lets the orchestrator distinguish recoverable conditions
(budget exhaustion, retrieval miss) from hard failures (ACL denial, citation
violation in strict mode).
"""

from __future__ import annotations


class KnowledgeAgentError(Exception):
    """Base class for all harness-specific errors."""


class RetrievalError(KnowledgeAgentError):
    """Raised when a retriever or the retrieval pipeline fails."""


class BudgetExceeded(KnowledgeAgentError):
    """Raised when an operation would exceed the remaining budget (SPEC §6.11)."""

    def __init__(self, requested: float, remaining: float) -> None:
        self.requested = requested
        self.remaining = remaining
        super().__init__(
            f"Budget exceeded: requested {requested:.4f} but only {remaining:.4f} remaining"
        )


class CitationViolation(KnowledgeAgentError):
    """Raised in strict mode when a claim lacks valid backing (SPEC §6.13)."""


class ACLDenied(KnowledgeAgentError):
    """Raised when a principal lacks read access to a chunk (SPEC §7.5)."""


class ConfigError(KnowledgeAgentError):
    """Raised on invalid or missing configuration."""


__all__ = [
    "KnowledgeAgentError",
    "RetrievalError",
    "BudgetExceeded",
    "CitationViolation",
    "ACLDenied",
    "ConfigError",
]
