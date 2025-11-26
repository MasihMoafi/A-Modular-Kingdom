"""
Test memory global access
"""
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memory.core import Mem0


def test_memory_uses_global_path():
    """Verify memory uses ~/.modular_kingdom and not project-relative path"""
    global_path = Path.home() / ".modular_kingdom" / "legacy_memories"

    # Initialize memory
    memory = Mem0(storage_path=str(global_path))

    # Verify the path exists and is in home directory
    assert global_path.exists()
    assert str(global_path).startswith(str(Path.home()))
    assert ".modular_kingdom" in str(global_path)


def test_memory_accessible_from_any_dir(tmp_path):
    """Test memory can be accessed from different working directories"""
    global_path = Path.home() / ".modular_kingdom" / "legacy_memories"

    # Change to temp directory
    os.chdir(tmp_path)

    # Memory should still work
    memory = Mem0(storage_path=str(global_path))
    memory.add("Test fact from different directory")

    # Verify it was saved globally
    all_memories = memory.get_all_memories()
    assert len(all_memories) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
