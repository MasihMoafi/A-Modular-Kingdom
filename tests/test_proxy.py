import os
import pytest
from src.utils.proxy import strip_all, strip_socks, _PROXY_KEYS, _SOCKS_KEYS

def test_strip_all(monkeypatch):
    """Test that strip_all removes all proxy environment variables."""
    # Set dummy values for all proxy keys and one non-proxy key
    for key in _PROXY_KEYS:
        monkeypatch.setenv(key, f"dummy_{key}")
    monkeypatch.setenv("NON_PROXY_KEY", "dummy_non_proxy")

    strip_all()

    # Check that all proxy keys are removed
    for key in _PROXY_KEYS:
        assert key not in os.environ

    # Check that non-proxy key remains
    assert "NON_PROXY_KEY" in os.environ
    assert os.environ["NON_PROXY_KEY"] == "dummy_non_proxy"


def test_strip_socks(monkeypatch):
    """Test that strip_socks removes only SOCKS/ALL_PROXY variables."""
    # Set dummy values for all proxy keys and one non-proxy key
    for key in _PROXY_KEYS:
        monkeypatch.setenv(key, f"dummy_{key}")
    monkeypatch.setenv("NON_PROXY_KEY", "dummy_non_proxy")

    strip_socks()

    # Check that SOCKS keys are removed
    for key in _SOCKS_KEYS:
        assert key not in os.environ

    # Check that other proxy keys remain
    for key in _PROXY_KEYS:
        if key not in _SOCKS_KEYS:
            assert key in os.environ
            assert os.environ[key] == f"dummy_{key}"

    # Check that non-proxy key remains
    assert "NON_PROXY_KEY" in os.environ
    assert os.environ["NON_PROXY_KEY"] == "dummy_non_proxy"
