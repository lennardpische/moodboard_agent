from __future__ import annotations

import hashlib
import random

from .schemas import CandidateImage, SourcePlan, StyleBrief


SOURCE_GOALS = {
    "Pinterest": "Find broad taste references, adjacent visual language, and clusters a director may respond to.",
    "ShotDeck": "Find cinematic frames with useful lighting, composition, lens, and color information.",
    "Web Image Search": "Find public references tied to studios, artists, titles, and production design terms.",
    "Manual Example Expansion": "Expand from user-provided examples into visually similar search neighborhoods.",
}


def build_source_plan(brief: StyleBrief, examples: list[str]) -> list[SourcePlan]:
    plans = [
        SourcePlan("Pinterest", brief.search_queries[:4], SOURCE_GOALS["Pinterest"], "mock"),
        SourcePlan("ShotDeck", brief.search_queries[1:5], SOURCE_GOALS["ShotDeck"], "mock"),
        SourcePlan("Web Image Search", brief.search_queries, SOURCE_GOALS["Web Image Search"], "mock"),
    ]
    if examples:
        plans.append(
            SourcePlan(
                "Manual Example Expansion",
                brief.search_queries[-2:],
                SOURCE_GOALS["Manual Example Expansion"],
                "mock",
            )
        )
    return plans


def collect_candidates(brief: StyleBrief, plans: list[SourcePlan], per_source: int = 24) -> list[CandidateImage]:
    candidates: list[CandidateImage] = []
    for plan in plans:
        for idx in range(per_source):
            candidates.append(_make_candidate(brief, plan, idx))
    return candidates


def _make_candidate(brief: StyleBrief, plan: SourcePlan, idx: int) -> CandidateImage:
    seed = f"{brief.directive}|{plan.source}|{idx}"
    rng = random.Random(seed)
    tags = _pick_tags(brief, rng)
    title = _title_from_tags(plan.source, tags, idx)
    image_seed = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    slug = "+".join(tags[:4])

    return CandidateImage(
        id=f"img_{image_seed}",
        title=title,
        source=plan.source,
        source_url=f"https://example.com/search/{plan.source.lower().replace(' ', '-')}/{slug}",
        thumbnail_url=f"https://picsum.photos/seed/{image_seed}/640/420",
        tags=tags,
        notes=_candidate_notes(plan.source, tags),
        rights_risk=rng.choice(["low", "medium", "medium", "unknown"]),
    )


def _pick_tags(brief: StyleBrief, rng: random.Random) -> list[str]:
    pools = [
        brief.keywords,
        brief.palette,
        brief.lighting,
        brief.composition,
        brief.texture,
    ]
    tags = []
    for pool in pools:
        if pool:
            tags.append(rng.choice(pool))
    rng.shuffle(tags)
    return tags


def _title_from_tags(source: str, tags: list[str], idx: int) -> str:
    lead = tags[0].title() if tags else "Reference"
    return f"{lead} study {idx + 1:02d} ({source})"


def _candidate_notes(source: str, tags: list[str]) -> str:
    if source == "ShotDeck":
        return f"Useful for cinematic language: {', '.join(tags[:3])}."
    if source == "Pinterest":
        return f"Useful for taste clustering and adjacent references: {', '.join(tags[:3])}."
    if source == "Manual Example Expansion":
        return f"Candidate derived from seed-example neighborhood: {', '.join(tags[:3])}."
    return f"General reference candidate: {', '.join(tags[:3])}."

