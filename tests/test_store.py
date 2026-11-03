from datetime import date

import pytest
from pydantic import ValidationError

from agentic_rag.schema import KnowledgeBase, StructuredQuery
from agentic_rag.store import KnowledgeStore


def source(tool="cursor"):
    return {"tool": tool, "project": "p", "file": "a.md", "anchor": "", "line_range": [1, 2]}


@pytest.fixture
def store():
    knowledge = KnowledgeBase()
    knowledge.items["rules"] = [
        {"id": "r1", "rule": "RTL רק במסכים ראשיים", "topic": "rtl", "tags": [], "observed_at": "2026-08-05", "source": source()},
        {"id": "r2", "rule": "RTL בכל המסכים", "topic": "rtl", "tags": [], "observed_at": "2026-10-05", "source": source("kiro")},
    ]
    knowledge.items["warnings"] = [
        {"id": "w1", "area": "loans", "message": "לא לגעת", "severity": "high", "topic": "state", "tags": [], "observed_at": "2026-10-30", "source": source()},
        {"id": "w2", "area": "misc", "message": "undated", "severity": "low", "topic": "x", "tags": [], "observed_at": "", "source": source()},
    ]
    knowledge.items["changes"] = [
        {"id": "c1", "area": "loans", "description": "added due_date", "change_type": "added", "topic": "loans", "tags": [], "observed_at": "2026-10-14", "source": source()},
    ]
    return KnowledgeStore(knowledge)


def ids(result):
    return [item["id"] for item in result[0]]


def test_no_filter_returns_everything_newest_first(store):
    assert ids(store.query(StructuredQuery()))[:2] == ["w1", "c1"]
    assert store.query(StructuredQuery())[1] == 5


def test_latest_per_topic_keeps_only_the_newest_rule(store):
    assert ids(store.query(StructuredQuery(item_types=["rules"], latest_per_topic=True))) == ["r2"]


def test_date_range_is_a_hard_filter_and_excludes_undated(store):
    query = StructuredQuery(item_types=["warnings"], since=date(2026, 10, 27), until=date(2026, 11, 3))
    assert ids(store.query(query)) == ["w1"]
    assert ids(store.query(StructuredQuery(since=date(2026, 11, 1)))) == []


def test_keywords_match_any_and_ignore_case(store):
    assert set(ids(store.query(StructuredQuery(keywords=["rtl", "ימין לשמאל"])))) == {"r1", "r2"}


def test_filters_by_tool_severity_and_change_type(store):
    assert ids(store.query(StructuredQuery(item_types=["rules"], tools=["kiro"]))) == ["r2"]
    assert ids(store.query(StructuredQuery(severity=["high"]))) == ["w1"]
    assert ids(store.query(StructuredQuery(change_types=["added"]))) == ["c1"]


def test_limit_reports_the_true_total(store):
    items, total = store.query(StructuredQuery(limit=1))
    assert len(items) == 1 and total == 5


def test_invalid_queries_are_rejected():
    with pytest.raises(ValidationError):
        StructuredQuery(item_types=["passwords"])
    with pytest.raises(ValidationError):
        StructuredQuery(since=date(2026, 2, 1), until=date(2026, 1, 1))


def test_save_and_load_round_trip_keeps_hebrew(store, tmp_path):
    path = tmp_path / "k" / "knowledge.json"
    store.save(path)
    assert "לא לגעת" in path.read_text(encoding="utf-8")
    assert KnowledgeStore.load(path).total_items == 5
    assert KnowledgeStore.load(tmp_path / "missing.json").total_items == 0
