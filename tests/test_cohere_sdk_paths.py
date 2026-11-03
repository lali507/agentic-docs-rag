"""Run the real Cohere SDK against a local fake API, to check the requests that are really sent."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.cohere import CohereEmbedding

from agentic_rag.cohere_llm import CohereChatLLM
from agentic_rag.config import Settings
from agentic_rag.llm_utils import predict_model
from agentic_rag.query_workflow import QueryWorkflow
from agentic_rag.schema import ExtractedSection, KnowledgeBase, SourceGroup
from agentic_rag.store import KnowledgeStore


@pytest.fixture
def api():
    calls: list[tuple[str, dict]] = []
    state = {"route": 1, "status": 200}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            calls.append((self.path, body))
            if "embed" in self.path:
                count = len(body["texts"])
                out = {"id": "x", "texts": body["texts"], "embeddings": {"float": [[0.1] * 1024] * count}, "meta": {}}
            else:
                prompt = body["messages"][0]["content"]
                if "Using only the choices above" in prompt:
                    text = json.dumps([{"choice": state["route"], "reason": "r"}])
                elif prompt.startswith("Translate the user"):
                    text = '```json\n{"item_types": ["warnings"]}\n```'
                elif "EXTRACT" in prompt:
                    text = '{"decisions": [{"title": "t", "summary": "s", "topic": "x"}]}'
                else:
                    text = "ANSWER"
                out = {
                    "id": "x", "finish_reason": "COMPLETE",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
                    "usage": {"tokens": {"input_tokens": 1, "output_tokens": 1}, "billed_units": {"input_tokens": 1, "output_tokens": 1}},
                }
            data = json.dumps(out).encode()
            self.send_response(state["status"])
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", calls, state
    server.shutdown()


async def test_completions_use_the_chat_api_and_not_the_deprecated_generate_api(api):
    url, calls, _ = api
    llm = CohereChatLLM(api_key="x", model="command-a-03-2025", base_url=url)
    assert (await llm.acomplete("hello")).text == "ANSWER"
    assert llm.complete("hello").text == "ANSWER"
    assert {path for path, _ in calls} == {"/v2/chat"}
    assert calls[0][1]["messages"] == [{"role": "user", "content": "hello"}]
    assert calls[0][1]["model"] == "command-a-03-2025"


async def test_structured_prediction_works_through_the_sdk(api):
    url, _, _ = api
    llm = CohereChatLLM(api_key="x", model="m", base_url=url)
    extracted = await predict_model(llm, ExtractedSection, "EXTRACT please")
    assert extracted.decisions[0].topic == "x"


async def test_query_embedding_sends_the_query_input_type(api):
    url, calls, _ = api
    embed = CohereEmbedding(api_key="x", model_name="embed-multilingual-v3.0", input_type="search_query", base_url=url)
    assert len(await embed.aget_query_embedding("שאלה")) == 1024
    path, body = calls[-1]
    assert "embed" in path and body["input_type"] == "search_query"
    assert body["model"] == "embed-multilingual-v3.0"


@pytest.mark.parametrize("route,expected", [(1, "semantic"), (2, "structured")])
async def test_whole_workflow_runs_through_the_sdk_on_both_routes(api, route, expected):
    url, calls, state = api
    state["route"] = route

    class Retriever:
        async def aretrieve(self, query):
            node = TextNode(text="body", metadata={"tool": "kiro", "file_path": "a.md"})
            return [NodeWithScore(node=node, score=0.9)]

    knowledge = KnowledgeBase(sources=[SourceGroup(tool="kiro", project="p")])
    knowledge.items["warnings"] = [{
        "id": "w", "area": "a", "message": "m", "severity": "high", "topic": "t", "tags": [],
        "observed_at": "2026-10-30", "source": {"tool": "kiro", "file": "a.md"},
    }]
    llm = CohereChatLLM(api_key="x", model="m", base_url=url)
    result = await QueryWorkflow(Settings(), llm, Retriever(), KnowledgeStore(knowledge)).run(question="שאלה")
    assert result.route == expected and result.answer == "ANSWER"
    assert {path for path, _ in calls} == {"/v2/chat"}


async def test_rate_limit_from_the_real_sdk_is_recognized(api):
    from agentic_rag.llm_utils import is_rate_limit_error

    url, _, state = api
    state["status"] = 429
    llm = CohereChatLLM(api_key="x", model="m", base_url=url)
    with pytest.raises(Exception) as caught:
        await llm.acomplete("hello")
    assert is_rate_limit_error(caught.value)
