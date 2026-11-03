"""Reads and queries the structured knowledge saved as a JSON file."""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from .schema import ITEM_TYPES, KnowledgeBase, StructuredQuery

_SEARCHABLE_FIELDS = (
    "title", "summary", "rule", "scope", "notes", "area", "message", "description", "topic",
)


def _searchable_text(item: dict) -> str:
    parts = [str(item.get(field, "")) for field in _SEARCHABLE_FIELDS]
    parts.extend(item.get("tags", []))
    return " ".join(parts).casefold()


def _item_date(item: dict) -> date | None:
    try:
        return date.fromisoformat(str(item.get("observed_at", ""))[:10])
    except ValueError:
        return None


class KnowledgeStore:
    def __init__(self, knowledge: KnowledgeBase | None = None):
        self.knowledge = knowledge or KnowledgeBase()

    @classmethod
    def load(cls, path: Path) -> "KnowledgeStore":
        if not path.exists():
            return cls()
        return cls(KnowledgeBase.model_validate_json(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.knowledge.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @property
    def total_items(self) -> int:
        return sum(len(items) for items in self.knowledge.items.values())

    def query(self, query: StructuredQuery) -> tuple[list[dict], int]:
        """Return the matching items, newest first, and the total before the limit."""
        selected: list[dict] = []
        for item_type in query.item_types or ITEM_TYPES:
            for item in self.knowledge.items.get(item_type, []):
                if self._matches(item, query):
                    selected.append({**item, "type": item_type})
        selected.sort(key=lambda item: _item_date(item) or date.min, reverse=True)
        if query.latest_per_topic:
            selected = self._latest_per_topic(selected)
        return selected[: query.limit], len(selected)

    @staticmethod
    def _matches(item: dict, query: StructuredQuery) -> bool:
        if query.tools and item.get("source", {}).get("tool") not in query.tools:
            return False
        if query.severity and item.get("severity") not in query.severity:
            return False
        if query.change_types and item.get("change_type") not in query.change_types:
            return False
        if query.keywords:
            text = _searchable_text(item)
            if not any(keyword.casefold() in text for keyword in query.keywords):
                return False
        if query.is_time_bound:
            # An undated item cannot honestly be said to fall inside a date range.
            observed = _item_date(item)
            if observed is None:
                return False
            if query.since and observed < query.since:
                return False
            if query.until and observed > query.until:
                return False
        return True

    @staticmethod
    def _latest_per_topic(newest_first: list[dict]) -> list[dict]:
        seen: set[tuple[str, str]] = set()
        latest = []
        for item in newest_first:
            key = (item["type"], item.get("topic") or item.get("id", ""))
            if key not in seen:
                seen.add(key)
                latest.append(item)
        return latest
