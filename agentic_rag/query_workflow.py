"""The event-driven question flow. Every step has one clear input and output event.

Events carry what the next step needs. ctx.store holds what spans the whole run.
docs/design-discussion.md explains these choices.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Optional

from llama_index.core import get_response_synthesizer
from llama_index.core.llms import LLM
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import ToolMetadata
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel, Field

from .config import Settings
from .llm_utils import predict_model, with_rate_limit_retry
from .prompts import (
    ANSWER_QA_PROMPT,
    ANSWER_REFINE_PROMPT,
    REWRITE_PROMPT,
    ROUTE_SEMANTIC,
    ROUTE_STRUCTURED,
    STRUCTURED_ANSWER_PROMPT,
    STRUCTURED_QUERY_PROMPT,
    UNTRUSTED_NOTE,
)
from .schema import StructuredQuery
from .store import KnowledgeStore

_HEBREW = re.compile(r"[֐-׿]")
SEMANTIC = "semantic"
STRUCTURED = "structured"


def _localized(question: str, hebrew: str, english: str) -> str:
    return hebrew if _HEBREW.search(question) else english


class AnswerResult(BaseModel):
    answer: str
    route: Optional[str] = None
    sources: list[str] = Field(default_factory=list)
    low_confidence: bool = False
    trace: list[str] = Field(default_factory=list)


# ---- Events ----
class QuestionAccepted(Event):
    question: str


class SemanticSearchRequested(Event):
    query: str
    # Results from earlier attempts, so that a poor rewrite cannot lose a good result.
    carried: list[NodeWithScore] = Field(default_factory=list)


class StructuredSearchRequested(Event):
    question: str


class NodesRetrieved(Event):
    query: str
    nodes: list[NodeWithScore]


class RewriteRequested(Event):
    best_nodes: list[NodeWithScore]
    reason: str


class SemanticAnswerRequested(Event):
    nodes: list[NodeWithScore]
    low_confidence: bool


class StructuredQueryReady(Event):
    query: StructuredQuery


class StructuredItemsFound(Event):
    found: list[dict]
    total: int


class NoAnswer(Event):
    reason_he: str
    reason_en: str


# ---- Postprocessing ----
class SourceHeaderPostprocessor(BaseNodePostprocessor):
    """Prefix each chunk with where it came from, so the LLM can cite and compare by date."""

    @classmethod
    def class_name(cls) -> str:
        return "SourceHeaderPostprocessor"

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> list[NodeWithScore]:
        formatted = []
        for item in nodes:
            metadata = item.node.metadata
            header = (
                f"[{metadata.get('tool', '?')} | {metadata.get('file_path', '?')} | "
                f"{metadata.get('section') or metadata.get('title', '')} | "
                f"{str(metadata.get('last_modified', ''))[:10]}]"
            )
            node = TextNode(text=f"{header}\n{item.node.get_content()}", id_=item.node.node_id)
            node.metadata = {}
            formatted.append(NodeWithScore(node=node, score=item.score))
        return formatted


def _citation(metadata: dict) -> str:
    start, end = int(metadata.get("line_start") or 0), int(metadata.get("line_end") or 0)
    lines = f":{start}-{end}" if start else ""
    return f"{metadata.get('tool', '?')} · {metadata.get('file_path', '?')}{lines}"


def _item_citation(item: dict) -> str:
    source = item.get("source", {})
    start, end = (source.get("line_range") or [0, 0])[:2]
    lines = f":{start}-{end}" if start else ""
    return f"{source.get('tool', '?')} · {source.get('file', '?')}{lines}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _merge_best(*groups: list[NodeWithScore]) -> list[NodeWithScore]:
    """Combine result lists, keeping each chunk once with its best score."""
    best: dict[str, NodeWithScore] = {}
    for group in groups:
        for item in group:
            current = best.get(item.node.node_id)
            if current is None or (item.score or 0) > (current.score or 0):
                best[item.node.node_id] = item
    return sorted(best.values(), key=lambda item: item.score or 0, reverse=True)


class QueryWorkflow(Workflow):
    def __init__(
        self,
        settings: Settings,
        llm: LLM,
        retriever,
        store: KnowledgeStore,
        today: date | None = None,
        **kwargs,
    ):
        kwargs.setdefault("timeout", 240)
        super().__init__(**kwargs)
        self.settings = settings
        self.llm = llm
        self.retriever = retriever
        self.store = store
        self.today = today
        self.selector = LLMSingleSelector.from_defaults(llm=llm)
        self.score_filter = SimilarityPostprocessor(similarity_cutoff=settings.min_score)
        self.formatter = SourceHeaderPostprocessor()
        self.synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=PromptTemplate(ANSWER_QA_PROMPT),
            refine_template=PromptTemplate(ANSWER_REFINE_PROMPT),
        )

    async def _note(self, ctx: Context, message: str) -> None:
        trace = await ctx.store.get("trace", default=[])
        await ctx.store.set("trace", [*trace, message])

    async def _finish(self, ctx: Context, answer: str, sources: list[str], low: bool = False) -> StopEvent:
        return StopEvent(
            result=AnswerResult(
                answer=answer,
                route=await ctx.store.get("route", default=None),
                sources=_unique(sources),
                low_confidence=low,
                trace=await ctx.store.get("trace", default=[]),
            )
        )

    # ---- No LLM: input validation ----
    @step
    async def validate_input(self, ctx: Context, ev: StartEvent) -> QuestionAccepted | StopEvent:
        question = str(ev.get("question") or "").strip()
        await ctx.store.set("question", question)
        await ctx.store.set("queries_tried", [])
        await ctx.store.set("trace", [])
        await ctx.store.set("route", None)
        if not question:
            return await self._finish(ctx, "נא לכתוב שאלה. / Please enter a question.", [])
        if len(question) > self.settings.max_question_chars:
            return await self._finish(
                ctx,
                _localized(
                    question,
                    f"השאלה ארוכה מדי (עד {self.settings.max_question_chars} תווים).",
                    f"The question is too long (limit {self.settings.max_question_chars} characters).",
                ),
                [],
            )
        return QuestionAccepted(question=question)

    # ---- LLM: choose the kind of search ----
    @step
    async def route(
        self, ctx: Context, ev: QuestionAccepted
    ) -> SemanticSearchRequested | StructuredSearchRequested:
        route = SEMANTIC
        if self.store.total_items == 0:
            await self._note(ctx, "Router skipped: no structured data, using semantic search")
        else:
            choices = [
                ToolMetadata(name=SEMANTIC, description=ROUTE_SEMANTIC),
                ToolMetadata(name=STRUCTURED, description=ROUTE_STRUCTURED),
            ]
            try:
                result = await with_rate_limit_retry(
                    lambda: self.selector.aselect(choices, query=ev.question)
                )
                route = STRUCTURED if result.selections[0].index == 1 else SEMANTIC
                await self._note(ctx, f"Router chose: {route}")
            except Exception as error:
                await self._note(ctx, f"Router failed ({type(error).__name__}), using semantic search")
        await ctx.store.set("route", route)
        if route == STRUCTURED:
            return StructuredSearchRequested(question=ev.question)
        return SemanticSearchRequested(query=ev.question)

    # ---- No LLM: vector search ----
    @step
    async def retrieve(self, ctx: Context, ev: SemanticSearchRequested) -> NodesRetrieved:
        tried = await ctx.store.get("queries_tried", default=[])
        await ctx.store.set("queries_tried", [*tried, ev.query])
        nodes = await self.retriever.aretrieve(ev.query)
        await self._note(ctx, f"Vector search found {len(nodes)} chunks for: {ev.query}")
        return NodesRetrieved(query=ev.query, nodes=_merge_best(nodes, ev.carried))

    # ---- No LLM: decide whether the results are good enough ----
    @step
    async def assess_results(
        self, ctx: Context, ev: NodesRetrieved
    ) -> SemanticAnswerRequested | RewriteRequested | NoAnswer:
        kept = self.score_filter.postprocess_nodes(ev.nodes)
        top_score = max((item.score or 0.0 for item in kept), default=0.0)
        attempts = len(await ctx.store.get("queries_tried", default=[]))
        confident = top_score >= self.settings.confidence_threshold
        await self._note(ctx, f"Assessed: {len(kept)} chunks kept, top score {top_score:.2f}")
        if confident:
            return SemanticAnswerRequested(nodes=kept, low_confidence=False)
        if attempts < self.settings.max_search_attempts:
            return RewriteRequested(best_nodes=ev.nodes, reason=f"top score {top_score:.2f}")
        if kept:
            return SemanticAnswerRequested(nodes=kept, low_confidence=True)
        return NoAnswer(
            reason_he="לא מצאתי במסמכים מידע שעונה על השאלה.",
            reason_en="I could not find information answering this in the documents.",
        )

    # ---- LLM: search again from another angle ----
    @step
    async def rewrite_query(self, ctx: Context, ev: RewriteRequested) -> SemanticSearchRequested | NoAnswer | SemanticAnswerRequested:
        question = await ctx.store.get("question")
        tried = await ctx.store.get("queries_tried", default=[])
        prompt = REWRITE_PROMPT.format(question=question, tried="; ".join(tried))
        rewritten = (
            await with_rate_limit_retry(lambda: self.llm.acomplete(prompt))
        ).text.strip().strip('"')
        if rewritten and rewritten not in tried:
            await self._note(ctx, f"Weak results ({ev.reason}), searching again")
            return SemanticSearchRequested(query=rewritten, carried=ev.best_nodes)
        kept = self.score_filter.postprocess_nodes(ev.best_nodes)
        if kept:
            return SemanticAnswerRequested(nodes=kept, low_confidence=True)
        return NoAnswer(
            reason_he="לא מצאתי במסמכים מידע שעונה על השאלה.",
            reason_en="I could not find information answering this in the documents.",
        )

    # ---- LLM: write the answer from chunks ----
    @step
    async def answer_semantic(self, ctx: Context, ev: SemanticAnswerRequested) -> StopEvent:
        question = await ctx.store.get("question")
        formatted = self.formatter.postprocess_nodes(ev.nodes)
        response = await with_rate_limit_retry(
            lambda: self.synthesizer.asynthesize(question, nodes=formatted)
        )
        answer = str(response).strip()
        if ev.low_confidence:
            warning = _localized(
                question,
                "⚠️ ההתאמה למסמכים נמוכה, ולכן כדאי לבדוק את התשובה מול המקורות.",
                "⚠️ The match with the documents is weak, so check this against the sources.",
            )
            answer = f"{warning}\n\n{answer}"
        sources = [_citation(item.node.metadata) for item in ev.nodes]
        return await self._finish(ctx, answer, sources, low=ev.low_confidence)

    # ---- LLM: translate the question into a query ----
    @step
    async def build_structured_query(
        self, ctx: Context, ev: StructuredSearchRequested
    ) -> StructuredQueryReady | SemanticSearchRequested:
        tools = sorted({group.tool for group in self.store.knowledge.sources})
        prompt = STRUCTURED_QUERY_PROMPT.format(
            today=(self.today or date.today()).isoformat(),
            tools=", ".join(tools) or "unknown",
            question=ev.question,
        )
        try:
            query = await predict_model(self.llm, StructuredQuery, prompt)
        except Exception as error:
            await self._note(ctx, f"Query generation failed ({type(error).__name__}), using semantic search")
            await ctx.store.set("route", SEMANTIC)
            return SemanticSearchRequested(query=ev.question)
        await self._note(ctx, "Query: " + query.model_dump_json(exclude_defaults=True))
        return StructuredQueryReady(query=query)

    # ---- No LLM: run the query and decide what an empty result means ----
    @step
    async def run_structured_query(
        self, ctx: Context, ev: StructuredQueryReady
    ) -> StructuredItemsFound | SemanticSearchRequested | NoAnswer:
        items, total = self.store.query(ev.query)
        await self._note(ctx, f"Structured query found {total} items")
        if items:
            return StructuredItemsFound(found=items, total=total)
        if ev.query.is_time_bound:
            # Falling back to similarity would return items outside the requested dates.
            return NoAnswer(
                reason_he="לא נמצאו פריטים תואמים בטווח הזמן שביקשת.",
                reason_en="No matching items were found in the requested time range.",
            )
        await ctx.store.set("route", SEMANTIC)
        return SemanticSearchRequested(query=await ctx.store.get("question"))

    # ---- LLM: write the answer from items ----
    @step
    async def answer_structured(self, ctx: Context, ev: StructuredItemsFound) -> StopEvent:
        question = await ctx.store.get("question")
        for_llm = [{k: v for k, v in item.items() if k != "id"} for item in ev.found]
        note = ""
        if ev.total > len(ev.found):
            note = f"Only {len(ev.found)} of {ev.total} items are shown. Tell the user this.\n"
        prompt = STRUCTURED_ANSWER_PROMPT.format(
            untrusted_note=UNTRUSTED_NOTE,
            truncation_note=note,
            question=question,
            items=json.dumps(for_llm, ensure_ascii=False, indent=1),
        )
        response = await with_rate_limit_retry(lambda: self.llm.acomplete(prompt))
        return await self._finish(
            ctx, response.text.strip(), [_item_citation(item) for item in ev.found]
        )

    # ---- No LLM: an honest "not found" ----
    @step
    async def no_answer(self, ctx: Context, ev: NoAnswer) -> StopEvent:
        question = await ctx.store.get("question")
        return await self._finish(ctx, _localized(question, ev.reason_he, ev.reason_en), [])
