import os
import importlib
from unittest.mock import patch
import pytest

# We need to test src.utils.proxy
# The module captures os.environ at import time, so we need to test this behavior

@pytest.fixture
def proxy_env():
    # Setup a mock environment
    env = {
        "HTTP_PROXY": "http://127.0.0.1:8080",
        "HTTPS_PROXY": "http://127.0.0.1:8080",
        "ALL_PROXY": "socks5://127.0.0.1:1080",
        "all_proxy": "socks5://127.0.0.1:1080",
        "OTHER_VAR": "value"
    }
    with patch.dict(os.environ, env, clear=True):
        # We need to reload the module so it captures the patched os.environ
        import src.utils.proxy as proxy
        importlib.reload(proxy)
        yield proxy

    # Teardown: reload the module again outside the patch
    # to restore its _original to the real os.environ
    import src.utils.proxy as proxy
    importlib.reload(proxy)

def test_initial_snapshot(proxy_env):
    """Test that the snapshot correctly captures only proxy variables."""
    assert proxy_env._original == {
        "HTTP_PROXY": "http://127.0.0.1:8080",
        "HTTPS_PROXY": "http://127.0.0.1:8080",
        "ALL_PROXY": "socks5://127.0.0.1:1080",
        "all_proxy": "socks5://127.0.0.1:1080",
    }

def test_strip_socks(proxy_env):
    """Test that SOCKS proxies are stripped but others are kept."""
    proxy_env.strip_socks()

    # SOCKS should be removed
    assert "ALL_PROXY" not in os.environ
    assert "all_proxy" not in os.environ

    # HTTP and other vars should be kept
    assert os.environ.get("HTTP_PROXY") == "http://127.0.0.1:8080"
    assert os.environ.get("HTTPS_PROXY") == "http://127.0.0.1:8080"
    assert os.environ.get("OTHER_VAR") == "value"

def test_restore_http(proxy_env):
    """Test that HTTP proxies are restored but not SOCKS."""
    # First clear everything to simulate an environment where proxies were removed
    os.environ.clear()

    proxy_env.restore_http()

    # HTTP should be restored
    assert os.environ.get("HTTP_PROXY") == "http://127.0.0.1:8080"
    assert os.environ.get("HTTPS_PROXY") == "http://127.0.0.1:8080"

    # SOCKS should NOT be restored
    assert "ALL_PROXY" not in os.environ
    assert "all_proxy" not in os.environ

    # Non-proxy vars were not in snapshot, shouldn't be restored
    assert "OTHER_VAR" not in os.environ

def test_strip_socks_when_not_set(proxy_env):
    """Test strip_socks when variables aren't set doesn't crash."""
    os.environ.clear()
    # Should not raise KeyError
    proxy_env.strip_socks()
