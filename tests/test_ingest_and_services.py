import json
from unittest.mock import MagicMock, patch

from agentic_rag import services
from agentic_rag.config import Settings
from agentic_rag.ingest_workflow import IngestWorkflow
from agentic_rag.store import KnowledgeStore
from fakes import FakeLLM, HashEmbedding

import pytest

EXTRACTED = json.dumps({
    "decisions": [{"title": "t", "summary": "s", "topic": "x", "tags": []}],
    "rules": [], "warnings": [], "changes": [],
})


def settings(tmp_path, **extra):
    return Settings(
        knowledge_path=tmp_path / "knowledge.json",
        extraction_cache_path=tmp_path / "cache.json",
        extraction_delay_seconds=0,
        **extra,
    )


async def test_full_preparation_on_the_sample_project(tmp_path):
    written = []
    config = settings(tmp_path)
    workflow = IngestWorkflow(
        config, FakeLLM(handler=lambda _: EXTRACTED),
        write_vectors=lambda chunks: (written.extend(chunks), len(chunks))[1],
    )
    result = await workflow.run()
    assert result["ok"] and result["files"] == 11 and result["vectors"] == len(written) > 0
    assert result["warnings"] == [] and result["extraction"]["failed"] == 0
    store = KnowledgeStore.load(config.knowledge_path)
    assert store.total_items == result["sections"] == result["items"]["decisions"]
    assert {group.tool for group in store.knowledge.sources} == {"claude_code", "cursor", "kiro"}


async def test_no_documents_stops_before_any_service_is_called(tmp_path):
    write = MagicMock()
    result = await IngestWorkflow(settings(tmp_path), None, write).run(roots=[tmp_path / "none"])
    assert not result["ok"] and "No tool documentation" in result["error"]
    write.assert_not_called()


async def test_single_tool_produces_a_warning_and_skips_extraction_without_llm(tmp_path):
    (tmp_path / "project" / ".kiro").mkdir(parents=True)
    (tmp_path / "project" / ".kiro" / "a.md").write_text("# A\n\n## B\n\nSome content here.\n", encoding="utf-8")
    result = await IngestWorkflow(settings(tmp_path), None, lambda c: len(c)).run(roots=[tmp_path / "project"])
    assert "at least two" in result["warnings"][0] and result["extraction"] == "skipped"


def test_embedding_input_types_differ_between_documents_and_queries():
    config = Settings(cohere_api_key="fake")
    assert services.document_embedding(config).input_type == "search_document"
    assert services.query_embedding(config).input_type == "search_query"
    assert services.document_embedding(config).model_name == "embed-multilingual-v3.0"


def test_configured_chat_model_is_accepted_by_the_library():
    assert services.get_llm(Settings(cohere_api_key="fake")).model == Settings().llm_model


def test_missing_keys_give_a_clear_error():
    with pytest.raises(services.MissingKeyError, match="COHERE_API_KEY"):
        services.get_llm(Settings())
    with pytest.raises(services.MissingKeyError, match="PINECONE_API_KEY"):
        services.get_vector_store(Settings())


def pinecone_mock(exists=True):
    client = MagicMock()
    client.has_index.return_value = exists
    client.Index.return_value.upsert.return_value = MagicMock(errors=[])
    return client


def test_rebuild_creates_a_missing_index_clears_it_and_upserts_with_metadata(tmp_path):
    from agentic_rag.chunking import chunk_sections, parse_sections
    from agentic_rag.sources import discover_files, load_documents

    config = Settings(pinecone_api_key="fake", cohere_api_key="fake")
    documents = load_documents(discover_files(config.docs_roots))
    chunks = chunk_sections(parse_sections(documents), documents, 500, 50)
    client = pinecone_mock(exists=False)
    index = client.Index.return_value
    with patch.object(services, "Pinecone", return_value=client), patch.object(
        services, "document_embedding", return_value=HashEmbedding()
    ):
        assert services.rebuild_vector_index(config, chunks) == len(chunks)
    assert client.create_index.call_args.kwargs["dimension"] == 1024
    index.delete.assert_called_once_with(namespace="agentic-docs", delete_all=True)
    vectors = [v for call in index.upsert.call_args_list for v in call.kwargs["vectors"]]
    assert len(vectors) == len(chunks) and len(vectors[0]["values"]) == 1024
    metadata = vectors[0]["metadata"]
    assert {"tool", "file_path", "title", "section", "line_start"} <= set(metadata)
    # What is written must rebuild into the same text when it is read back.
    from llama_index.core.vector_stores.utils import metadata_dict_to_node

    assert metadata_dict_to_node(metadata).get_content() == chunks[0].get_content()
    assert all(call.kwargs["namespace"] == "agentic-docs" for call in index.upsert.call_args_list)


def test_clearing_a_namespace_that_does_not_exist_is_fine():
    from llama_index.core.schema import TextNode

    config = Settings(pinecone_api_key="fake")
    client = pinecone_mock()
    client.Index.return_value.delete.side_effect = RuntimeError("Namespace not found")
    with patch.object(services, "Pinecone", return_value=client), patch.object(
        services, "document_embedding", return_value=HashEmbedding()
    ):
        assert services.rebuild_vector_index(config, [TextNode(text="x")]) == 1


def test_other_clear_errors_are_not_hidden():
    config = Settings(pinecone_api_key="fake")
    client = pinecone_mock()
    client.Index.return_value.delete.side_effect = RuntimeError("Unauthorized")
    with patch.object(services, "Pinecone", return_value=client), pytest.raises(RuntimeError):
        services.rebuild_vector_index(config, [])


async def test_retriever_reads_pinecone_results_including_metadata_and_scores():
    from types import SimpleNamespace

    config = Settings(pinecone_api_key="fake", cohere_api_key="fake")
    match = SimpleNamespace(
        id="n1",
        score=0.81,
        metadata={"text": "the body", "tool": "kiro", "file_path": "a.md", "line_start": 3},
        values=[],
    )
    client = pinecone_mock()
    index = client.Index.return_value
    index.query.return_value = SimpleNamespace(matches=[match])
    with patch.object(services, "Pinecone", return_value=client), patch.object(
        services, "query_embedding", return_value=HashEmbedding()
    ):
        nodes = await services.get_retriever(config).aretrieve("שאלה")
    assert index.query.call_args.kwargs["namespace"] == "agentic-docs"
    assert index.query.call_args.kwargs["top_k"] == config.top_k
    assert nodes[0].score == 0.81
    assert nodes[0].node.get_content() == "the body"
    assert nodes[0].node.metadata["tool"] == "kiro"


def test_openai_key_selects_openai_models(monkeypatch):
    from agentic_rag import config as config_module

    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)
    for name in ("LLM_PROVIDER", "PINECONE_INDEX", "OPENAI_LLM_MODEL", "EMBED_MODEL", "EMBED_DIMENSION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    loaded = config_module.load_settings()
    assert loaded.llm_provider == "openai" and loaded.embed_dimension == 1536
    assert loaded.pinecone_index == "agentic-docs-openai" and loaded.extraction_delay_seconds == 0
    assert services.get_llm(loaded).model == loaded.llm_model
    assert services.query_embedding(loaded).model_name == "text-embedding-3-small"


def test_missing_openai_key_gives_a_clear_error():
    with pytest.raises(services.MissingKeyError, match="OPENAI_API_KEY"):
        services.get_llm(Settings(llm_provider="openai"))


async def test_local_vector_store_round_trip_without_pinecone(tmp_path):
    from llama_index.core.schema import TextNode

    config = Settings(local_vector_dir=tmp_path / "vectors")
    nodes = [TextNode(text="the database is postgres", metadata={"tool": "kiro"}),
             TextNode(text="buttons are blue", metadata={"tool": "cursor"})]
    with patch.object(services, "document_embedding", return_value=HashEmbedding()), patch.object(
        services, "query_embedding", return_value=HashEmbedding()
    ):
        assert services.rebuild_vector_index(config, nodes) == 2
        found = await services.get_retriever(config).aretrieve("which database postgres")
    assert found[0].node.get_content() == "the database is postgres"
    assert found[0].node.metadata["tool"] == "kiro"


def test_local_retriever_before_prepare_explains_what_to_run(tmp_path):
    with pytest.raises(services.MissingKeyError, match="prepare"):
        services.get_retriever(Settings(local_vector_dir=tmp_path / "none"))
