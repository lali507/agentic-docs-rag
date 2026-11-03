from datetime import date
from pathlib import Path

from agentic_rag.chunking import chunk_sections, parse_sections
from agentic_rag.config import Settings
from agentic_rag.sources import discover_files, load_documents

ROOTS = Settings().docs_roots


def load():
    documents = load_documents(discover_files(ROOTS))
    return documents, parse_sections(documents)


def test_finds_hidden_folders_and_mdc_files_for_each_tool():
    files = discover_files(ROOTS)
    assert {file.tool for file in files} == {"claude_code", "cursor", "kiro"}
    assert ".cursor/rules/ui.mdc" in {file.relative_path for file in files}


def test_missing_root_is_ignored():
    assert discover_files([Path("/no/such/folder")]) == []


def test_date_comes_from_front_matter_not_the_filesystem():
    documents, _ = load()
    by_path = {document.metadata["file_path"]: document for document in documents}
    assert by_path["CLAUDE.md"].metadata["last_modified"].startswith("2026-10-20")


def test_front_matter_and_empty_headings_are_not_sections():
    _, sections = load()
    assert sections
    for section in sections:
        text = section.get_content()
        assert not text.startswith("---")
        assert any(line.strip() and not line.startswith("#") for line in text.splitlines())


def test_line_ranges_point_at_the_real_text():
    documents, sections = load()
    chunks = chunk_sections(sections, documents, 500, 50)
    texts = {document.metadata["file_path"]: document.text.split("\n") for document in documents}
    assert len(chunks) >= len(sections)
    for chunk in chunks:
        metadata = chunk.metadata
        assert metadata["line_start"] > 0, metadata
        slice_ = "\n".join(texts[metadata["file_path"]][metadata["line_start"] - 1 : metadata["line_end"]])
        assert chunk.get_content() in slice_


def test_long_section_is_split_with_overlap_and_correct_lines(tmp_path):
    (tmp_path / ".kiro").mkdir()
    body = "\n".join(f"Sentence number {i} talks about loans and books." for i in range(300))
    (tmp_path / ".kiro" / "long.md").write_text(f"# Title\n\n## Long\n\n{body}\n", encoding="utf-8")
    documents = load_documents(discover_files([tmp_path]))
    chunks = chunk_sections(parse_sections(documents), documents, 120, 20)
    assert len(chunks) > 2
    assert chunks[0].metadata["line_start"] < chunks[1].metadata["line_start"]
    assert all(chunk.metadata["line_start"] > 0 for chunk in chunks)


def test_metadata_is_flat_and_embedding_text_excludes_noise():
    documents, sections = load()
    chunk = chunk_sections(sections, documents, 500, 50)[0]
    assert all(isinstance(value, (str, int, float)) for value in chunk.metadata.values())
    from llama_index.core.schema import MetadataMode

    embedded = chunk.get_content(metadata_mode=MetadataMode.EMBED)
    assert "sha256" not in embedded and chunk.metadata["tool"] in embedded
