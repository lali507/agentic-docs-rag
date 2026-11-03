"""Event-driven preparation: load -> chunk -> index -> extract -> save."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from llama_index.core.llms import LLM
from llama_index.core.schema import TextNode
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

from .chunking import chunk_sections, parse_sections
from .config import Settings
from .extraction import ExtractionCache, extract_knowledge
from .schema import KnowledgeBase
from .sources import discover_files, load_documents
from .store import KnowledgeStore

_MIN_TOOLS = 2


class DocumentsLoaded(Event):
    documents: list
    tools: list[str]
    warnings: list[str]


class ChunksReady(Event):
    documents: list
    sections: list[TextNode]
    chunks: list[TextNode]
    warnings: list[str]


class VectorsIndexed(Event):
    documents: list
    sections: list[TextNode]
    summary: dict


class ItemsExtracted(Event):
    knowledge: KnowledgeBase
    summary: dict


class IngestWorkflow(Workflow):
    def __init__(
        self,
        settings: Settings,
        llm: LLM | None,
        write_vectors: Callable[[list[TextNode]], int],
        **kwargs,
    ):
        kwargs.setdefault("timeout", None)
        super().__init__(**kwargs)
        self.settings = settings
        self.llm = llm
        self.write_vectors = write_vectors

    @step
    async def load(self, ev: StartEvent) -> DocumentsLoaded | StopEvent:
        roots: list[Path] = list(ev.get("roots") or self.settings.docs_roots)
        files = discover_files(roots)
        if not files:
            shown = ", ".join(str(root) for root in roots)
            return StopEvent(result={"ok": False, "error": f"No tool documentation found in: {shown}"})
        tools = sorted({file.tool for file in files})
        warnings = []
        if len(tools) < _MIN_TOOLS:
            warnings.append(f"Only one tool found ({tools[0]}); the assignment asks for at least two.")
        documents = [d for d in load_documents(files) if d.text.strip()]
        if not documents:
            return StopEvent(result={"ok": False, "error": "All documentation files are empty."})
        return DocumentsLoaded(documents=documents, tools=tools, warnings=warnings)

    @step
    async def chunk(self, ev: DocumentsLoaded) -> ChunksReady:
        sections = parse_sections(ev.documents)
        chunks = chunk_sections(
            sections, ev.documents, self.settings.chunk_size, self.settings.chunk_overlap
        )
        return ChunksReady(
            documents=ev.documents, sections=sections, chunks=chunks, warnings=ev.warnings
        )

    @step
    async def index_vectors(self, ev: ChunksReady) -> VectorsIndexed:
        written = self.write_vectors(ev.chunks)
        summary = {
            "ok": True,
            "files": len(ev.documents),
            "sections": len(ev.sections),
            "vectors": written,
            "warnings": ev.warnings,
        }
        return VectorsIndexed(documents=ev.documents, sections=ev.sections, summary=summary)

    @step
    async def extract(self, ev: VectorsIndexed) -> ItemsExtracted | StopEvent:
        if self.llm is None:
            return StopEvent(result={**ev.summary, "extraction": "skipped"})
        knowledge, stats = await extract_knowledge(
            ev.documents,
            ev.sections,
            self.llm,
            ExtractionCache(self.settings.extraction_cache_path),
            self.settings.llm_model,
            self.settings.extraction_delay_seconds,
        )
        return ItemsExtracted(knowledge=knowledge, summary={**ev.summary, "extraction": stats})

    @step
    async def save(self, ev: ItemsExtracted) -> StopEvent:
        store = KnowledgeStore(ev.knowledge)
        store.save(self.settings.knowledge_path)
        counts = {name: len(items) for name, items in ev.knowledge.items.items()}
        return StopEvent(result={**ev.summary, "items": counts})
