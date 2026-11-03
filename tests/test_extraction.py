import json

from agentic_rag.chunking import parse_sections
from agentic_rag.extraction import ExtractionCache, extract_knowledge
from agentic_rag.sources import discover_files, load_documents
from fakes import FakeLLM


def docs(tmp_path):
    folder = tmp_path / ".kiro"
    folder.mkdir()
    (folder / "a.md").write_text(
        "---\nupdated: 2026-10-01\n---\n# A\n\n## DB choice\n\nWe chose SQLite for the first stage.\n",
        encoding="utf-8",
    )
    documents = load_documents(discover_files([tmp_path]))
    return documents, parse_sections(documents)


GOOD = json.dumps({
    "decisions": [{
        "title": "DB", "summary": "SQLite", "topic": "database", "tags": ["db"],
        "observed_at": None, "source": {"file": "invented.md"}, "id": "invented",
    }],
    "rules": [], "warnings": [], "changes": [],
})


async def run(tmp_path, llm, cache=None):
    documents, sections = docs(tmp_path) if not (tmp_path / ".kiro").exists() else (
        load_documents(discover_files([tmp_path])),
        parse_sections(load_documents(discover_files([tmp_path]))),
    )
    cache = cache or ExtractionCache(tmp_path / "cache.json")
    return await extract_knowledge(documents, sections, llm, cache, "fake"), cache


async def test_source_comes_from_the_file_and_never_from_the_llm(tmp_path):
    (knowledge, stats), _ = await run(tmp_path, FakeLLM(handler=lambda _: GOOD))
    item = knowledge.items["decisions"][0]
    assert stats["extracted"] == 1
    assert item["source"]["file"] == ".kiro/a.md"
    assert item["source"]["tool"] == "kiro"
    assert item["source"]["anchor"] == "DB choice"
    assert item["source"]["line_range"][0] == 6
    assert item["id"].startswith("dec-") and item["id"] != "invented"
    assert item["observed_at"] == "2026-10-01"
    assert knowledge.sources[0].files[0].hash.startswith("sha256:")


async def test_second_run_uses_the_cache_and_makes_no_llm_call(tmp_path):
    first = FakeLLM(handler=lambda _: GOOD)
    (_, _), cache = await run(tmp_path, first)
    second = FakeLLM(handler=lambda _: GOOD)
    (knowledge, stats), _ = await run(tmp_path, second, ExtractionCache(cache.path))
    assert second.prompts == []
    assert stats["from_cache"] == 1 and len(knowledge.items["decisions"]) == 1


async def test_bad_llm_output_is_counted_and_does_not_stop_the_run(tmp_path):
    (knowledge, stats), cache = await run(tmp_path, FakeLLM(handler=lambda _: "not json"))
    assert stats["failed"] == 1 and sum(map(len, knowledge.items.values())) == 0
    assert cache.entries == {}


async def test_fenced_json_is_accepted(tmp_path):
    (knowledge, _), _ = await run(tmp_path, FakeLLM(handler=lambda _: f"```json\n{GOOD}\n```"))
    assert len(knowledge.items["decisions"]) == 1


async def test_date_written_in_the_section_wins_and_nonsense_is_ignored(tmp_path):
    def with_date(value):
        data = json.loads(GOOD)
        data["decisions"][0]["observed_at"] = value
        return json.dumps(data)

    (knowledge, _), _ = await run(tmp_path, FakeLLM(handler=lambda _: with_date("2026-07-10")))
    assert knowledge.items["decisions"][0]["observed_at"] == "2026-07-10"
    (tmp_path / "cache.json").unlink()
    (knowledge, _), _ = await run(tmp_path, FakeLLM(handler=lambda _: with_date("last summer")))
    assert knowledge.items["decisions"][0]["observed_at"] == "2026-10-01"
