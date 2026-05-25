"""Tests for typed settings (SPEC §10.5)."""

from common.settings import Settings, get_settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("QDRANT_URL", "http://example:6333")
    # _env_file=None so the real .env (if any) does not interfere with the test.
    s = Settings(_env_file=None)
    assert s.anthropic_api_key == "sk-test"
    assert s.qdrant_url == "http://example:6333"
    assert s.langsmith_project == "knowledge-agent"


def test_settings_defaults_without_env(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.openai_api_key is None
    assert s.qdrant_url == "http://localhost:6333"


def test_get_settings_is_cached():
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
