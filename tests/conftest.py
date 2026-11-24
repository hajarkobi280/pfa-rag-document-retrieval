"""Offline unit tests must never probe a running Ollama for status metadata."""
import pytest

@pytest.fixture(autouse=True)
def offline_status(request, monkeypatch):
    if request.node.get_closest_marker('slow'):
        return
    import app
    monkeypatch.setattr(app, 'ollama_version', lambda: 'mock-version')
    app._retrieve_details_cached.cache_clear()
