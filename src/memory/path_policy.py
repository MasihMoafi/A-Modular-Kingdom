import os
from pathlib import Path


def resolve_memory_base(project_root: str | None = None) -> str:
    """Single policy seam for choosing the runtime memory base path."""
    explicit = os.getenv("MEMORY_BASE_PATH", "").strip()
    if explicit:
        return str(Path(explicit).expanduser())

    # Default global location for cross-project persistence.
    return str((Path.home() / ".elpis" / "memories").expanduser())
