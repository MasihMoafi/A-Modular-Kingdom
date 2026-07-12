"""
Markdown-backed memory mirror and lexical index.

This is intentionally small: markdown files are the inspectable source of truth.
Vector search can be layered on later, but this store must work alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from pathlib import Path
from typing import Iterable

from memory.memory_config import MemoryConfig, MemoryScope


ENTRY_RE = re.compile(
    r"^<!-- memory-id:(?P<id>[^ ]+) scope:(?P<scope>[^ ]+) created:(?P<created>[^ ]+) -->$"
)
END_RE = re.compile(r"^<!-- /memory-id:(?P<id>[^ ]+) -->$")


@dataclass
class MarkdownMemoryEntry:
    id: str
    scope: str
    content: str
    source: str
    start_line: int
    end_line: int


class MarkdownMemoryStore:
    """File-first memory store used as a mirror/search source."""

    def __init__(self, config: MemoryConfig):
        self.config = config

    def append(self, memory_id: str, content: str, scope: MemoryScope) -> dict:
        path = self._path_for_scope(scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file_header(path)

        created = datetime.now().isoformat(timespec="seconds")
        content_lines = content.rstrip().splitlines() or [""]
        block = [
            f"<!-- memory-id:{memory_id} scope:{scope.value} created:{created} -->",
            f"### {scope.value}",
            "",
            *content_lines,
            "",
            f"<!-- /memory-id:{memory_id} -->",
            "",
        ]

        existing_lines = path.read_text(encoding="utf-8").splitlines()
        start_line = len(existing_lines) + 1
        with path.open("a", encoding="utf-8") as f:
            f.write("\n".join(block))
            f.write("\n")

        return {
            "source": str(path),
            "start_line": start_line,
            "end_line": start_line + len(block) - 1,
        }

    def delete(self, memory_id: str) -> bool:
        deleted = False
        for path in self._candidate_files():
            lines = path.read_text(encoding="utf-8").splitlines()
            new_lines: list[str] = []
            idx = 0
            changed = False
            while idx < len(lines):
                match = ENTRY_RE.match(lines[idx])
                if match and match.group("id") == memory_id:
                    idx += 1
                    while idx < len(lines):
                        end = END_RE.match(lines[idx])
                        idx += 1
                        if end and end.group("id") == memory_id:
                            break
                    while idx < len(lines) and lines[idx] == "":
                        idx += 1
                    changed = True
                    deleted = True
                    continue
                new_lines.append(lines[idx])
                idx += 1
            if changed:
                path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
        return deleted

    def get(self, memory_id: str) -> MarkdownMemoryEntry | None:
        for entry in self._entries():
            if entry.id == memory_id:
                return entry
        return None

    def list_all(self, scope: MemoryScope) -> list[dict]:
        return [self._entry_to_dict(entry) for entry in self._entries([scope])]

    def storage_files(self) -> list[str]:
        return [str(path) for path in self._candidate_files()]

    def search(
        self,
        query: str,
        k: int = 3,
        scopes: Iterable[MemoryScope] | None = None,
    ) -> list[dict]:
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []

        scored: list[tuple[float, MarkdownMemoryEntry]] = []
        for entry in self._entries(scopes):
            content_tokens = set(self._tokenize(entry.content))
            if not content_tokens:
                continue
            hits = query_tokens & content_tokens
            if not hits:
                continue
            score = len(hits) / max(len(query_tokens), 1)
            scored.append((score, entry))

        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:k]
        return [self._entry_to_dict(entry, text_score=score) for score, entry in ranked]

    def _entry_to_dict(self, entry: MarkdownMemoryEntry, text_score: float = 1.0) -> dict:
        return {
            "id": entry.id,
            "content": entry.content,
            "metadata": {
                "scope": entry.scope,
                "source": entry.source,
                "start_line": entry.start_line,
                "end_line": entry.end_line,
                "score": text_score,
                "text_score": text_score,
                "vector_score": None,
            },
        }

    def _entries(self, scopes: Iterable[MemoryScope] | None = None) -> list[MarkdownMemoryEntry]:
        wanted = {scope.value for scope in scopes} if scopes else None
        entries: list[MarkdownMemoryEntry] = []
        for path in self._candidate_files():
            lines = path.read_text(encoding="utf-8").splitlines()
            idx = 0
            while idx < len(lines):
                match = ENTRY_RE.match(lines[idx])
                if not match:
                    idx += 1
                    continue

                memory_id = match.group("id")
                scope = match.group("scope")
                start_line = idx + 1
                body: list[str] = []
                idx += 1
                while idx < len(lines):
                    end = END_RE.match(lines[idx])
                    if end and end.group("id") == memory_id:
                        end_line = idx + 1
                        content = self._clean_body(body)
                        if content and (wanted is None or scope in wanted):
                            entries.append(
                                MarkdownMemoryEntry(
                                    id=memory_id,
                                    scope=scope,
                                    content=content,
                                    source=str(path),
                                    start_line=start_line,
                                    end_line=end_line,
                                )
                            )
                        break
                    body.append(lines[idx])
                    idx += 1
                idx += 1
        return entries

    def _candidate_files(self) -> list[Path]:
        roots = [
            self.config.global_memory_base / "global",
            self.config.global_memory_base / "projects" / self.config.project_hash,
        ]
        files: list[Path] = []
        for root in roots:
            memory_file = root / "MEMORY.md"
            if memory_file.exists():
                files.append(memory_file)
            daily_dir = root / "memory"
            if daily_dir.exists():
                files.extend(sorted(daily_dir.glob("*.md")))
        return files

    def _path_for_scope(self, scope: MemoryScope) -> Path:
        if scope == MemoryScope.PROJECT_SESSIONS:
            date_name = datetime.now().strftime("%Y-%m-%d.md")
            return self.config.global_memory_base / "projects" / self.config.project_hash / "memory" / date_name
        if scope.name.startswith("GLOBAL_"):
            return self.config.global_memory_base / "global" / "MEMORY.md"
        return self.config.global_memory_base / "projects" / self.config.project_hash / "MEMORY.md"

    def _ensure_file_header(self, path: Path) -> None:
        if path.exists():
            return
        title = "Project Memory" if "projects" in path.parts else "Global Memory"
        path.write_text(f"# {title}\n\n", encoding="utf-8")

    def _clean_body(self, lines: list[str]) -> str:
        cleaned = list(lines)
        if cleaned and cleaned[0].startswith("### "):
            cleaned = cleaned[1:]
        while cleaned and cleaned[0] == "":
            cleaned = cleaned[1:]
        while cleaned and cleaned[-1] == "":
            cleaned = cleaned[:-1]
        return "\n".join(cleaned).strip()

    def _tokenize(self, text: str) -> list[str]:
        return [tok for tok in re.split(r"[^a-z0-9]+", (text or "").lower()) if tok]
