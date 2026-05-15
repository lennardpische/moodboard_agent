from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MoodboardRequest:
    directive: str
    examples: list[str] = field(default_factory=list)
    target_count: int = 36

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StyleBrief:
    directive: str
    summary: str
    keywords: list[str]
    palette: list[str]
    lighting: list[str]
    composition: list[str]
    texture: list[str]
    negative_cues: list[str]
    search_queries: list[str]
    llm_used: bool = False
    llm_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreBreakdown:
    style_match: float
    visual_variety: float
    composition_value: float
    prompting_utility: float
    rights_safety: float
    total: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateImage:
    id: str
    title: str
    source: str
    source_url: str
    thumbnail_url: str
    tags: list[str]
    notes: str
    rights_risk: str
    score: ScoreBreakdown | None = None
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.score:
            payload["score"] = self.score.to_dict()
        return payload


@dataclass
class SourcePlan:
    source: str
    queries: list[str]
    goal: str
    adapter: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MoodboardRun:
    request: MoodboardRequest
    style_brief: StyleBrief
    source_plan: list[SourcePlan]
    selected_images: list[CandidateImage]
    rejected_images: list[CandidateImage]
    next_actions: list[str]
    id: str = field(default_factory=lambda: f"run_{uuid4().hex[:10]}")
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "request": self.request.to_dict(),
            "style_brief": self.style_brief.to_dict(),
            "source_plan": [plan.to_dict() for plan in self.source_plan],
            "selected_images": [image.to_dict() for image in self.selected_images],
            "rejected_images": [image.to_dict() for image in self.rejected_images],
            "next_actions": self.next_actions,
        }

