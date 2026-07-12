import os
from unittest.mock import patch
from src.utils.proxy import get_http_env

def test_get_http_env_includes_non_proxy_vars():
    """Test that get_http_env includes regular environment variables."""
    with patch.dict("os.environ", {"SOME_VAR": "value", "OTHER_VAR": "123"}, clear=True):
        with patch("src.utils.proxy._original", {}):
            env = get_http_env()
            assert env.get("SOME_VAR") == "value"
            assert env.get("OTHER_VAR") == "123"

def test_get_http_env_removes_socks_proxy():
    """Test that get_http_env removes ALL_PROXY and all_proxy."""
    with patch.dict("os.environ", {"ALL_PROXY": "socks5://127.0.0.1:1080", "all_proxy": "socks5://127.0.0.1:1080", "SOME_VAR": "value"}, clear=True):
        with patch("src.utils.proxy._original", {}):
            env = get_http_env()
            assert "ALL_PROXY" not in env
            assert "all_proxy" not in env
            assert env.get("SOME_VAR") == "value"

def test_get_http_env_restores_http_proxy_from_original():
    """Test that get_http_env adds back HTTP_PROXY, HTTPS_PROXY, FTP_PROXY from _original."""
    with patch.dict("os.environ", {"SOME_VAR": "value"}, clear=True):
        with patch("src.utils.proxy._original", {
            "HTTP_PROXY": "http://127.0.0.1:8080",
            "https_proxy": "http://127.0.0.1:8080",
            "ftp_proxy": "ftp://127.0.0.1:2121",
            "IGNORE_THIS": "should_not_be_restored"
        }):
            env = get_http_env()
            assert env.get("SOME_VAR") == "value"
            assert env.get("HTTP_PROXY") == "http://127.0.0.1:8080"
            assert env.get("https_proxy") == "http://127.0.0.1:8080"
            assert env.get("ftp_proxy") == "ftp://127.0.0.1:2121"
            assert "IGNORE_THIS" not in env

def test_get_http_env_does_not_mutate_os_environ():
    """Test that get_http_env doesn't change the actual os.environ."""
    with patch.dict("os.environ", {"ALL_PROXY": "socks5://127.0.0.1:1080", "SOME_VAR": "value"}, clear=True):
        with patch("src.utils.proxy._original", {"HTTP_PROXY": "http://127.0.0.1:8080"}):
            env = get_http_env()

            # The returned env is modified
            assert "ALL_PROXY" not in env
            assert env.get("HTTP_PROXY") == "http://127.0.0.1:8080"

            # The actual os.environ is untouched
            assert os.environ.get("ALL_PROXY") == "socks5://127.0.0.1:1080"
            assert "HTTP_PROXY" not in os.environ
            assert os.environ.get("SOME_VAR") == "value"
