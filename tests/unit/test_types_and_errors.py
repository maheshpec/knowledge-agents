"""Tests for common.types and common.errors."""

import pytest

from common.errors import BudgetExceeded, CitationViolation, KnowledgeAgentError
from common.types import BudgetGrant, BudgetSpec, LLMCallRecord, MemoryItem, MimeType, ToolCall


def test_mimetype_values():
    assert MimeType.PDF.value == "application/pdf"
    assert MimeType("text/markdown") is MimeType.MARKDOWN


def test_budget_grant_has_unique_id():
    g1 = BudgetGrant(amount=1.0)
    g2 = BudgetGrant(amount=1.0)
    assert g1.grant_id != g2.grant_id
    assert g1.settled is False


def test_budget_spec_defaults():
    spec = BudgetSpec()
    assert spec.max_usd == 1.0
    assert spec.max_tokens is None


def test_tool_call_autogenerates_id():
    tc = ToolCall(tool="search")
    assert tc.id
    assert tc.status == "pending"


def test_memory_item_timestamped():
    item = MemoryItem(key="k", value=123)
    assert item.scope == "working"
    assert item.created_at is not None


def test_llm_call_record_defaults():
    rec = LLMCallRecord(model="claude-sonnet-4-6")
    assert rec.cost_usd == 0.0
    assert rec.cache_hit is False


def test_budget_exceeded_is_knowledge_agent_error():
    err = BudgetExceeded(requested=5.0, remaining=1.0)
    assert isinstance(err, KnowledgeAgentError)
    assert err.requested == 5.0
    assert "5.0" in str(err)


def test_citation_violation_hierarchy():
    with pytest.raises(KnowledgeAgentError):
        raise CitationViolation("unsupported claim")
