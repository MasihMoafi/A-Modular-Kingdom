import os
import sys
from unittest.mock import MagicMock
import pytest

# Mock heavy ML dependencies before importing rag modules
sys.modules['src.rag.core'] = MagicMock()
sys.modules['src.rag.core_2'] = MagicMock()

from src.rag.fetch import resolve_path

def test_resolve_path_empty():
    assert resolve_path("") is None
    assert resolve_path(None) is None

def test_resolve_path_shortcuts(monkeypatch):
    monkeypatch.setenv("HOME", "/home/testuser")

    # Check shortcuts resolution
    assert resolve_path("desktop") == "/home/testuser/Desktop"

    # Check case insensitivity and whitespace handling
    assert resolve_path(" dOcUmEnTs ") == "/home/testuser/Documents"

    assert resolve_path("DOWNLOADS") == "/home/testuser/Downloads"

def test_resolve_path_expanduser(monkeypatch):
    monkeypatch.setenv("HOME", "/home/testuser")
    assert resolve_path("~/some_folder") == "/home/testuser/some_folder"

def test_resolve_path_expandvars(monkeypatch):
    monkeypatch.setenv("CUSTOM_TEST_DIR", "/opt/custom")
    assert resolve_path("$CUSTOM_TEST_DIR/logs") == "/opt/custom/logs"

def test_resolve_path_relative():
    cwd = os.getcwd()
    expected = os.path.join(cwd, "some/relative/path")
    assert resolve_path("some/relative/path") == expected

def test_resolve_path_absolute():
    abs_path = "/var/log/test.log"
    assert resolve_path(abs_path) == abs_path
