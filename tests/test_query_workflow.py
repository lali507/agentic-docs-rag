import json
from datetime import date

import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode

from agentic_rag.chunking import chunk_sections, parse_sections
from agentic_rag.config import Settings
from agentic_rag.query_workflow import QueryWorkflow
from agentic_rag.schema import KnowledgeBase, SourceGroup
from agentic_rag.sources import discover_files, load_documents
from agentic_rag.store import KnowledgeStore
from fakes import FakeLLM, HashEmbedding

SETTINGS = Settings(min_score=0.2, confidence_threshold=0.35)


def source():
    return {"tool": "kiro", "project": "p", "file": "x.md", "anchor": "h", "line_range": [3, 4]}


def filled_store():
    knowledge = KnowledgeBase(sources=[SourceGroup(tool="kiro", project="p")])
    knowledge.items["warnings"] = [
        {"id": "w1", "area": "loans", "message": "רגיש מאוד", "severity": "high", "topic": "state", "tags": [], "observed_at": "2026-10-30", "source": source()}
    ]
    return KnowledgeStore(knowledge)


class Handler:
    """Plays the LLM's roles, and identifies each role by its prompt."""

    def __init__(self, route=1, query=None, rewrite="another angle"):
        self.route, self.query, self.rewrite = route, query or {}, rewrite

    def __call__(self, prompt: str) -> str:
        if "Using only the choices above" in prompt:
            if self.route == "broken":
                return "no idea"
            return json.dumps([{"choice": self.route, "reason": "test"}])
        if prompt.startswith("Translate the user's question"):
            return self.query if isinstance(self.query, str) else json.dumps(self.query)
        if prompt.startswith("A search over project"):
            return self.rewrite
        if prompt.startswith("Answer the question using only the items"):
            return "STRUCTURED ANSWER"
        return "SEMANTIC ANSWER"


class ScriptedRetriever:
    def __init__(self, *rounds):
        self.rounds, self.queries = list(rounds), []

    async def aretrieve(self, query):
        self.queries.append(query)
        return self.rounds.pop(0) if self.rounds else []


def hit(score, node_id="n1"):
    node = TextNode(
        text="content",
        id_=node_id,
        metadata={"tool": "cursor", "file_path": "a.md", "line_start": 2, "line_end": 5, "section": "S", "last_modified": "2026-01-01"},
    )
    return NodeWithScore(node=node, score=score)


def workflow(handler, retriever, store=None):
    llm = FakeLLM(handler=handler)
    return (
        QueryWorkflow(SETTINGS, llm, retriever, store if store is not None else filled_store(), today=date(2026, 11, 3)),
        llm,
    )


@pytest.mark.parametrize("question", ["", "   ", None])
async def test_empty_question_makes_no_calls(question):
    retriever = ScriptedRetriever()
    wf, llm = workflow(Handler(), retriever)
    result = await wf.run(question=question)
    assert llm.prompts == [] and retriever.queries == [] and "Please enter" in result.answer


async def test_too_long_question_is_rejected():
    wf, llm = workflow(Handler(), ScriptedRetriever())
    result = await wf.run(question="א" * 3000)
    assert llm.prompts == [] and "ארוכה" in result.answer


async def test_semantic_route_answers_and_cites():
    wf, _ = workflow(Handler(route=1), ScriptedRetriever([hit(0.8)]))
    result = await wf.run(question="איך מתקינים?")
    assert result.route == "semantic" and result.answer == "SEMANTIC ANSWER"
    assert result.sources == ["cursor · a.md:2-5"] and not result.low_confidence


async def test_chunks_reach_the_llm_with_their_source_header():
    wf, llm = workflow(Handler(route=1), ScriptedRetriever([hit(0.8)]))
    await wf.run(question="q")
    assert "[cursor | a.md | S | 2026-01-01]" in llm.prompts[-1]


async def test_weak_result_triggers_one_rewrite_then_uses_the_better_result():
    retriever = ScriptedRetriever([hit(0.25, "weak")], [hit(0.7, "strong")])
    wf, _ = workflow(Handler(route=1), retriever)
    result = await wf.run(question="שאלה")
    assert retriever.queries == ["שאלה", "another angle"]
    assert result.answer == "SEMANTIC ANSWER" and not result.low_confidence


async def test_still_weak_after_retry_is_answered_with_a_warning_and_keeps_best_result():
    retriever = ScriptedRetriever([hit(0.3, "first")], [hit(0.22, "second")])
    wf, llm = workflow(Handler(route=1), retriever)
    result = await wf.run(question="שאלה")
    assert result.low_confidence and "⚠️" in result.answer and len(retriever.queries) == 2


async def test_nothing_found_says_so_without_inventing_an_answer():
    retriever = ScriptedRetriever([], [])
    wf, llm = workflow(Handler(route=1), retriever)
    result = await wf.run(question="שאלה")
    assert "לא מצאתי" in result.answer and result.sources == []
    assert not any("SEMANTIC" in p for p in llm.prompts)


async def test_rewrite_repeating_the_same_query_stops_the_loop():
    retriever = ScriptedRetriever([hit(0.25)])
    wf, _ = workflow(Handler(route=1, rewrite="שאלה"), retriever)
    result = await wf.run(question="שאלה")
    assert len(retriever.queries) == 1 and result.low_confidence


async def test_structured_route_answers_from_items():
    wf, llm = workflow(Handler(route=2, query={"item_types": ["warnings"]}), ScriptedRetriever())
    result = await wf.run(question="מה רגיש?")
    assert result.route == "structured" and result.answer == "STRUCTURED ANSWER"
    assert result.sources == ["kiro · x.md:3-4"] and "רגיש מאוד" in llm.prompts[-1]
    assert "2026-11-03" in next(p for p in llm.prompts if p.startswith("Translate"))


async def test_empty_time_bound_query_does_not_fall_back_to_similarity():
    retriever = ScriptedRetriever([hit(0.9)])
    wf, _ = workflow(Handler(route=2, query={"since": "2027-01-01"}), retriever)
    result = await wf.run(question="מה קרה בשבוע האחרון?")
    assert retriever.queries == [] and "בטווח הזמן" in result.answer


async def test_empty_query_without_dates_falls_back_to_semantic():
    retriever = ScriptedRetriever([hit(0.9)])
    wf, _ = workflow(Handler(route=2, query={"keywords": ["no such thing"]}), retriever)
    result = await wf.run(question="רשימה של משהו")
    assert retriever.queries == ["רשימה של משהו"] and result.route == "semantic"


@pytest.mark.parametrize("bad", ["not json", json.dumps({"item_types": ["passwords"]})])
async def test_malformed_query_falls_back_to_semantic(bad):
    retriever = ScriptedRetriever([hit(0.9)])
    wf, _ = workflow(Handler(route=2, query=bad), retriever)
    result = await wf.run(question="רשימה")
    assert result.route == "semantic" and result.answer == "SEMANTIC ANSWER"


async def test_broken_router_falls_back_to_semantic():
    retriever = ScriptedRetriever([hit(0.9)])
    wf, _ = workflow(Handler(route="broken"), retriever)
    result = await wf.run(question="שאלה")
    assert result.route == "semantic" and any("Router failed" in line for line in result.trace)


async def test_no_structured_data_skips_the_router():
    wf, llm = workflow(Handler(), ScriptedRetriever([hit(0.9)]), store=KnowledgeStore())
    result = await wf.run(question="שאלה")
    assert result.route == "semantic"
    assert not any("Using only the choices" in p for p in llm.prompts)


async def test_rate_limit_is_retried(monkeypatch):
    class TooManyRequestsError(Exception):
        pass

    calls = {"n": 0}

    def flaky(prompt):
        if prompt.startswith("Answer the question using only the items"):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TooManyRequestsError()
        return Handler(route=2, query={"item_types": ["warnings"]})(prompt)

    import agentic_rag.llm_utils as utils
    monkeypatch.setattr(utils, "_RATE_LIMIT_WAIT_SECONDS", 0.0)
    original = utils.with_rate_limit_retry

    async def no_wait(call, wait_seconds=0.0):
        return await original(call, 0.0)

    monkeypatch.setattr("agentic_rag.query_workflow.with_rate_limit_retry", no_wait)
    wf, _ = workflow(flaky, ScriptedRetriever())
    result = await wf.run(question="מה רגיש?")
    assert result.answer == "STRUCTURED ANSWER" and calls["n"] == 2


async def test_real_sample_data_reaches_the_right_chunk_through_a_real_index():
    documents = load_documents(discover_files(SETTINGS.docs_roots))
    chunks = chunk_sections(parse_sections(documents), documents, 500, 50)
    retriever = VectorStoreIndex(chunks, embed_model=HashEmbedding()).as_retriever(similarity_top_k=4)
    wf, llm = workflow(Handler(route=1), retriever, store=KnowledgeStore())
    result = await wf.run(question="איך מתקינים ומריצים pnpm install db:migrate")
    assert any("CLAUDE.md" in source for source in result.sources)
    assert "pnpm install" in llm.prompts[-1]


def test_exhausted_quota_is_not_treated_as_a_rate_limit():
    from agentic_rag.llm_utils import is_rate_limit_error

    class Fake429(Exception):
        status_code = 429

    assert is_rate_limit_error(Fake429("slow down"))
    assert not is_rate_limit_error(Fake429("Error code: 429 - {'type': 'insufficient_quota'}"))
