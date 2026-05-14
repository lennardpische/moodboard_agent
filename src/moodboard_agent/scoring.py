from __future__ import annotations

import hashlib

from .schemas import CandidateImage, ScoreBreakdown, StyleBrief


DEFAULT_WEIGHTS = {
    "style_match": 0.36,
    "visual_variety": 0.18,
    "composition_value": 0.16,
    "prompting_utility": 0.18,
    "rights_safety": 0.12,
}


def score_candidates(
    candidates: list[CandidateImage],
    brief: StyleBrief,
    weights: dict[str, float] | None = None,
) -> list[CandidateImage]:
    active_weights = weights or DEFAULT_WEIGHTS
    scored = []
    for candidate in candidates:
        candidate.score = _score_candidate(candidate, brief, active_weights)
        scored.append(candidate)
    return scored


def select_diverse_candidates(candidates: list[CandidateImage], target_count: int) -> tuple[list[CandidateImage], list[CandidateImage]]:
    sorted_candidates = sorted(candidates, key=lambda item: item.score.total if item.score else 0, reverse=True)
    selected: list[CandidateImage] = []
    seen_sources: dict[str, int] = {}
    seen_lead_tags: set[str] = set()

    for candidate in sorted_candidates:
        if len(selected) >= target_count:
            break

        lead_tag = candidate.tags[0] if candidate.tags else ""
        source_count = seen_sources.get(candidate.source, 0)
        source_limit = max(3, target_count // 2)
        is_varied = lead_tag not in seen_lead_tags or len(selected) < target_count // 2

        if source_count < source_limit and is_varied:
            candidate.selected = True
            selected.append(candidate)
            seen_sources[candidate.source] = source_count + 1
            seen_lead_tags.add(lead_tag)

    if len(selected) < target_count:
        selected_ids = {candidate.id for candidate in selected}
        for candidate in sorted_candidates:
            if len(selected) >= target_count:
                break
            if candidate.id not in selected_ids:
                candidate.selected = True
                selected.append(candidate)
                selected_ids.add(candidate.id)

    selected_ids = {candidate.id for candidate in selected}
    rejected = [candidate for candidate in sorted_candidates if candidate.id not in selected_ids]
    return selected, rejected


def _score_candidate(candidate: CandidateImage, brief: StyleBrief, weights: dict[str, float]) -> ScoreBreakdown:
    tag_text = " ".join(candidate.tags).lower()
    style_hits = sum(1 for keyword in brief.keywords if keyword.lower() in tag_text)
    composition_hits = sum(1 for term in brief.composition if term.lower() in tag_text)
    texture_hits = sum(1 for term in brief.texture if term.lower() in tag_text)

    style_match = min(1.0, 0.46 + style_hits * 0.16 + _stable_noise(candidate.id, "style") * 0.16)
    visual_variety = 0.52 + _stable_noise(candidate.id, "variety") * 0.38
    composition_value = min(1.0, 0.44 + composition_hits * 0.2 + _stable_noise(candidate.id, "composition") * 0.2)
    prompting_utility = min(1.0, 0.5 + texture_hits * 0.16 + len(candidate.tags) * 0.035)
    rights_safety = {"low": 0.86, "medium": 0.62, "unknown": 0.48}.get(candidate.rights_risk, 0.5)

    total = (
        style_match * weights["style_match"]
        + visual_variety * weights["visual_variety"]
        + composition_value * weights["composition_value"]
        + prompting_utility * weights["prompting_utility"]
        + rights_safety * weights["rights_safety"]
    )

    rationale = (
        f"Matches {style_hits} core keyword(s), has {composition_hits} composition cue(s), "
        f"and carries {candidate.rights_risk} rights risk."
    )

    return ScoreBreakdown(
        style_match=round(style_match, 3),
        visual_variety=round(visual_variety, 3),
        composition_value=round(composition_value, 3),
        prompting_utility=round(prompting_utility, 3),
        rights_safety=round(rights_safety, 3),
        total=round(total, 3),
        rationale=rationale,
    )


def _stable_noise(value: str, salt: str) -> float:
    digest = hashlib.sha1(f"{value}|{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF

