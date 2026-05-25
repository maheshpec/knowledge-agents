"""Integration smoke tests.

Phase 1A ships a placeholder so the integration suite collects cleanly in CI.
Tests requiring external services (Qdrant, network) must be marked
``@pytest.mark.integration`` and are skipped in the default CI lane.
"""

from common import Chunk, get_settings


def test_package_imports_and_settings_construct(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.langsmith_project == "knowledge-agent"
    chunk = Chunk(chunk_id="c", doc_id="d", text="hello")
    assert chunk.text == "hello"
