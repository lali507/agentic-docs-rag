"""Split documents along markdown structure, then to a size that fits embedding."""
from __future__ import annotations

import re

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode

_FRONT_MATTER_ONLY = re.compile(r"\A---\s*\n.*\n---\s*\Z", re.DOTALL)


def _document_key(metadata: dict) -> tuple[str, str]:
    return metadata["project"], metadata["file_path"]


def _annotate_location(nodes: list[TextNode], documents: list[Document]) -> None:
    """Record each node's line range so that answers can cite an exact place."""
    texts = {_document_key(document.metadata): document.text for document in documents}
    search_from: dict[tuple[str, str], int] = {}
    for node in nodes:
        key = _document_key(node.metadata)
        full_text = texts.get(key, "")
        body = node.get_content()
        start = full_text.find(body, search_from.get(key, 0))
        if start < 0:
            start = full_text.find(body)
        if start < 0:
            node.metadata["line_start"] = 0
            node.metadata["line_end"] = 0
            continue
        # Chunks can overlap, so the next search starts at this chunk's start, not its end.
        search_from[key] = start
        node.metadata["line_start"] = full_text.count("\n", 0, start) + 1
        node.metadata["line_end"] = full_text.count("\n", 0, start + len(body)) + 1


def _has_content(section: TextNode) -> bool:
    """False for front matter and for headings with nothing under them, which only add noise."""
    text = section.get_content().strip()
    if _FRONT_MATTER_ONLY.match(text):
        return False
    return any(line.strip() and not line.lstrip().startswith("#") for line in text.splitlines())


def _heading_of(section: TextNode) -> str:
    first_line = section.get_content().lstrip().split("\n", 1)[0]
    return first_line.lstrip("#").strip() if first_line.startswith("#") else ""


def parse_sections(documents: list[Document]) -> list[TextNode]:
    """Return one node per markdown section. This is also the unit used for extraction."""
    parsed = MarkdownNodeParser().get_nodes_from_documents(documents)
    sections = [section for section in parsed if _has_content(section)]
    _annotate_location(sections, documents)
    for section in sections:
        section.metadata["section"] = _heading_of(section)
        section.metadata.pop("header_path", None)
    return sections


def chunk_sections(
    sections: list[TextNode],
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextNode]:
    """Split long sections. Both sizes are measured in tokens."""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.get_nodes_from_documents(sections)
    _annotate_location(chunks, documents)
    return chunks
