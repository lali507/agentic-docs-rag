"""Find and load the markdown documentation kept by each agentic coding tool."""
from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader

# Plain text reading is all that is needed here, so the optional readers package stays out.
logging.getLogger("llama_index.core.readers.file.base").setLevel(logging.ERROR)

# Where each tool keeps its files. Adding a tool only requires adding a line here.
TOOL_PATTERNS: dict[str, tuple[str, ...]] = {
    "claude_code": ("CLAUDE.md", ".claude/**/*.md"),
    "cursor": (".cursor/**/*.md", ".cursor/**/*.mdc"),
    "kiro": (".kiro/**/*.md",),
    "copilot": (".github/copilot-instructions.md", ".github/instructions/**/*.md"),
    "codex": ("AGENTS.md",),
}

# Metadata that is useful for filtering and citing but only adds noise to an embedding.
NON_SEMANTIC_KEYS = [
    "file_hash", "last_modified", "last_modified_ts", "project", "line_start", "line_end",
]

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_DATE_FIELD = re.compile(
    r"^(?:updated|last_modified|last_updated|date)\s*:\s*[\"']?([0-9]{4}-[0-9]{2}-[0-9]{2}[^\"'\n]*)",
    re.MULTILINE | re.IGNORECASE,
)
_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class SourceFile:
    tool: str
    project: str
    root: Path
    path: Path

    @property
    def relative_path(self) -> str:
        return self.path.relative_to(self.root).as_posix()


def discover_files(roots: tuple[Path, ...] | list[Path]) -> list[SourceFile]:
    """Return every documentation file under the given project roots."""
    found: list[SourceFile] = []
    for root in roots:
        if not root.is_dir():
            continue
        for tool, patterns in TOOL_PATTERNS.items():
            paths = {path for pattern in patterns for path in root.glob(pattern) if path.is_file()}
            found.extend(
                SourceFile(tool=tool, project=root.name, root=root, path=path)
                for path in sorted(paths)
            )
    return found


def resolve_last_modified(path: Path, text: str) -> datetime:
    """Date a file: front matter first, then git history, then the filesystem.

    Filesystem dates alone are unreliable because a fresh clone resets them all to the
    clone time, which would break every time-based question.
    """
    front_matter = _FRONT_MATTER.match(text)
    if front_matter:
        field = _DATE_FIELD.search(front_matter.group(1))
        if field:
            parsed = parse_datetime(field.group(1))
            if parsed:
                return parsed
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", path.name],
            cwd=path.parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        parsed = parse_datetime(result.stdout.strip()) if result.returncode == 0 else None
        if parsed:
            return parsed
    except (OSError, subprocess.SubprocessError):
        pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def load_documents(files: list[SourceFile]) -> list[Document]:
    """Load files with LlamaIndex's reader and attach the context each one came from."""
    if not files:
        return []
    by_path = {str(file.path): file for file in files}
    # input_files is explicit because the reader skips hidden folders such as .kiro by default.
    reader = SimpleDirectoryReader(
        input_files=list(by_path),
        encoding="utf-8",
        exclude_hidden=False,
        raise_on_error=True,
    )
    documents = reader.load_data()
    for document in documents:
        file = by_path[str(Path(document.metadata["file_path"]))]
        modified = resolve_last_modified(file.path, document.text)
        heading = _H1.search(document.text)
        document.metadata = {
            "tool": file.tool,
            "project": file.project,
            "file_path": file.relative_path,
            "title": heading.group(1).strip() if heading else file.path.stem,
            "last_modified": modified.isoformat(),
            "last_modified_ts": int(modified.timestamp()),
            "file_hash": "sha256:" + hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
        }
        document.excluded_embed_metadata_keys = list(NON_SEMANTIC_KEYS)
        document.excluded_llm_metadata_keys = list(NON_SEMANTIC_KEYS)
    return documents
