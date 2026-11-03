"""Stage C data preparation: turn markdown sections into structured items."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from llama_index.core.llms import LLM
from llama_index.core.schema import TextNode

from .llm_utils import predict_model
from .prompts import EXTRACTION_PROMPT, EXTRACTION_PROMPT_VERSION, UNTRUSTED_NOTE
from .schema import ITEM_TYPES, ExtractedSection, ItemSource, KnowledgeBase, SourceFileEntry, SourceGroup

logger = logging.getLogger(__name__)
_MIN_SECTION_CHARS = 20
_ID_PREFIX = {"decisions": "dec", "rules": "rule", "warnings": "warn", "changes": "chg"}


class ExtractionCache:
    """Skip sections that did not change: the free key allows 1,000 calls a month."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict] = {}
        if path.exists():
            self.entries = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(section_text: str, model_name: str) -> str:
        material = f"{EXTRACTION_PROMPT_VERSION}|{model_name}|{section_text}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, ensure_ascii=False), encoding="utf-8")


def _valid_date_or_none(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10]).isoformat()
    except ValueError:
        return None


def _to_items(extracted: ExtractedSection, section: TextNode) -> dict[str, list[dict]]:
    """Attach source details from the file itself, never from the LLM."""
    metadata = section.metadata
    source = ItemSource(
        tool=metadata["tool"],
        project=metadata["project"],
        file=metadata["file_path"],
        anchor=metadata.get("section", ""),
        line_range=(metadata.get("line_start", 0), metadata.get("line_end", 0)),
    ).model_dump(mode="json")
    items: dict[str, list[dict]] = {}
    for item_type in ITEM_TYPES:
        items[item_type] = []
        for entry in getattr(extracted, item_type):
            item = entry.model_dump(mode="json")
            item["observed_at"] = (
                _valid_date_or_none(item.get("observed_at")) or metadata["last_modified"][:10]
            )
            content = json.dumps([item_type, source["file"], item], ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]
            items[item_type].append({"id": f"{_ID_PREFIX[item_type]}-{digest}", **item, "source": source})
    return items


def _source_groups(documents: list) -> list[SourceGroup]:
    groups: dict[tuple[str, str], SourceGroup] = {}
    for document in documents:
        metadata = document.metadata
        key = (metadata["tool"], metadata["project"])
        group = groups.setdefault(key, SourceGroup(tool=key[0], project=key[1]))
        group.files.append(
            SourceFileEntry(
                path=metadata["file_path"],
                last_modified=metadata["last_modified"],
                hash=metadata["file_hash"],
            )
        )
    return list(groups.values())


async def extract_knowledge(
    documents: list,
    sections: list[TextNode],
    llm: LLM,
    cache: ExtractionCache,
    model_name: str,
    delay_seconds: float = 0.0,
) -> tuple[KnowledgeBase, dict[str, int]]:
    """Extract all sections. One failed section is counted but does not stop the run."""
    knowledge = KnowledgeBase(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sources=_source_groups(documents),
    )
    stats = {"sections": 0, "from_cache": 0, "extracted": 0, "failed": 0}
    for section in sections:
        text = section.get_content().strip()
        if len(text) < _MIN_SECTION_CHARS:
            continue
        stats["sections"] += 1
        key = ExtractionCache.key(text, model_name)
        if key in cache.entries:
            extracted = ExtractedSection.model_validate(cache.entries[key])
            stats["from_cache"] += 1
        else:
            prompt = EXTRACTION_PROMPT.format(
                untrusted_note=UNTRUSTED_NOTE,
                file_path=section.metadata["file_path"],
                tool=section.metadata["tool"],
                section=section.metadata.get("section", ""),
                text=text,
            )
            try:
                extracted = await predict_model(llm, ExtractedSection, prompt)
            except Exception as error:
                logger.warning(
                    "Extraction failed for %s (%s): %s",
                    section.metadata["file_path"], section.metadata.get("section"), error,
                )
                stats["failed"] += 1
                continue
            cache.entries[key] = extracted.model_dump(mode="json")
            cache.save()
            stats["extracted"] += 1
            if delay_seconds:
                await asyncio.sleep(delay_seconds)
        for item_type, items in _to_items(extracted, section).items():
            knowledge.items[item_type].extend(items)
    return knowledge, stats
