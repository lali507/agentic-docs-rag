"""The structured layer: what is extracted from documents and how it can be queried."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.0"
ItemType = Literal["decisions", "rules", "warnings", "changes"]
ITEM_TYPES: tuple[str, ...] = ("decisions", "rules", "warnings", "changes")


# ---- What the LLM returns for one section. It never decides source, id or file dates. ----
class ExtractedDecision(BaseModel):
    title: str
    summary: str
    topic: str = Field(description="Short stable English slug, for example 'primary-color'")
    tags: list[str] = Field(default_factory=list)
    observed_at: Optional[str] = Field(None, description="YYYY-MM-DD only if stated in the text")


class ExtractedRule(BaseModel):
    rule: str
    scope: str = ""
    notes: str = ""
    topic: str = Field(description="Short stable English slug, for example 'rtl'")
    tags: list[str] = Field(default_factory=list)
    observed_at: Optional[str] = Field(None, description="YYYY-MM-DD only if stated in the text")


class ExtractedWarning(BaseModel):
    area: str
    message: str
    severity: Literal["low", "medium", "high"] = "medium"
    topic: str = Field(description="Short stable English slug, for example 'auth-hook'")
    tags: list[str] = Field(default_factory=list)
    observed_at: Optional[str] = Field(None, description="YYYY-MM-DD only if stated in the text")


class ExtractedChange(BaseModel):
    area: str = Field(description="What changed, for example 'db' or a table name")
    description: str
    change_type: Literal["added", "removed", "modified"] = "modified"
    topic: str = Field(description="Short stable English slug, for example 'loans-table'")
    tags: list[str] = Field(default_factory=list)
    observed_at: Optional[str] = Field(None, description="YYYY-MM-DD only if stated in the text")


class ExtractedSection(BaseModel):
    decisions: list[ExtractedDecision] = Field(default_factory=list)
    rules: list[ExtractedRule] = Field(default_factory=list)
    warnings: list[ExtractedWarning] = Field(default_factory=list)
    changes: list[ExtractedChange] = Field(default_factory=list)


# ---- The saved knowledge file. ----
class ItemSource(BaseModel):
    tool: str
    project: str
    file: str
    anchor: str = ""
    line_range: tuple[int, int] = (0, 0)


class SourceFileEntry(BaseModel):
    path: str
    last_modified: str
    hash: str


class SourceGroup(BaseModel):
    tool: str
    project: str
    files: list[SourceFileEntry] = Field(default_factory=list)


class KnowledgeBase(BaseModel):
    schema_version: str = SCHEMA_VERSION
    generated_at: str = ""
    sources: list[SourceGroup] = Field(default_factory=list)
    items: dict[str, list[dict]] = Field(
        default_factory=lambda: {item_type: [] for item_type in ITEM_TYPES}
    )


# ---- The query the LLM writes against that file. ----
class StructuredQuery(BaseModel):
    """Validation here is a guard: a malformed query never reaches the data."""

    item_types: list[ItemType] = Field(default_factory=list, description="Empty means all types")
    keywords: list[str] = Field(
        default_factory=list,
        description="An item must contain at least one. Give Hebrew and English variants.",
    )
    tools: list[str] = Field(default_factory=list, description="Empty means all tools")
    severity: list[Literal["low", "medium", "high"]] = Field(default_factory=list)
    change_types: list[Literal["added", "removed", "modified"]] = Field(default_factory=list)
    since: Optional[date] = None
    until: Optional[date] = None
    latest_per_topic: bool = Field(
        False, description="True when the user wants what is currently in force"
    )
    limit: int = Field(50, ge=1, le=200)

    @model_validator(mode="after")
    def _dates_in_order(self) -> "StructuredQuery":
        if self.since and self.until and self.since > self.until:
            raise ValueError("since must not be later than until")
        return self

    @property
    def is_time_bound(self) -> bool:
        return self.since is not None or self.until is not None
