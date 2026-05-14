from __future__ import annotations

import json
from pathlib import Path

from .schemas import MoodboardRequest, MoodboardRun
from .scoring import DEFAULT_WEIGHTS, score_candidates, select_diverse_candidates
from .sources import build_source_plan, collect_candidates
from .style_analysis import analyze_style


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "pipeline_config.json"


def run_moodboard_pipeline(
    directive: str,
    examples: list[str] | None = None,
    target_count: int | None = None,
) -> MoodboardRun:
    config = _load_config()
    request = MoodboardRequest(
        directive=directive,
        examples=examples or [],
        target_count=target_count or int(config.get("default_target_count", 36)),
    )

    style_brief = analyze_style(request)
    source_plan = build_source_plan(style_brief, request.examples)
    candidates = collect_candidates(
        style_brief,
        source_plan,
        per_source=int(config.get("max_candidates_per_source", 24)),
    )
    scored = score_candidates(candidates, style_brief, config.get("scoring_weights", DEFAULT_WEIGHTS))
    selected, rejected = select_diverse_candidates(scored, request.target_count)

    return MoodboardRun(
        request=request,
        style_brief=style_brief,
        source_plan=source_plan,
        selected_images=selected,
        rejected_images=rejected,
        next_actions=_next_actions(),
    )


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _next_actions() -> list[str]:
    return [
        "Review selected references and pin the strongest 8-12 images.",
        "Reject off-style references so the agent can learn the boundary of the brief.",
        "Cluster accepted images into character, environment, lighting, palette, and texture groups.",
        "Replace mock source adapters with real Pinterest, ShotDeck, and web-search adapters.",
        "Prepare a Midjourney handoff queue after human approval.",
    ]

