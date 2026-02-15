"""Central proxy manager — single source of truth for proxy handling."""
import os

_PROXY_KEYS = [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "ftp_proxy",
]
_SOCKS_KEYS = ["ALL_PROXY", "all_proxy"]

# Snapshot at import time (before anything strips them)
_original = {k: os.environ[k] for k in _PROXY_KEYS if k in os.environ}


def strip_all():
    """Remove every proxy var. Use for local-only services (Qdrant local)."""
    for k in _PROXY_KEYS:
        os.environ.pop(k, None)


def strip_socks():
    """Remove only SOCKS/ALL_PROXY. Keep HTTP(S) for cloud services."""
    for k in _SOCKS_KEYS:
        os.environ.pop(k, None)


def restore_http():
    """Restore only HTTP(S) proxy. Safe for processes that need cloud access but break on SOCKS."""
    for k, v in _original.items():
        if k.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY"):
            os.environ[k] = v


def restore_all():
    """Restore every saved proxy var."""
    for k, v in _original.items():
        os.environ[k] = v


def get_http_env():
    """Return env dict with only HTTP(S) proxy — for subprocess.Popen(env=...)."""
    env = os.environ.copy()
    for k in _SOCKS_KEYS:
        env.pop(k, None)
    for k, v in _original.items():
        if k.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY"):
            env[k] = v
    return env
