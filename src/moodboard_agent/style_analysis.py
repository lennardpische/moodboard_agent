from __future__ import annotations

import re

from .schemas import MoodboardRequest, StyleBrief


STYLE_LIBRARY = {
    "pixar": {
        "keywords": ["stylized 3d animation", "expressive character design", "cinematic family adventure", "appealing shapes"],
        "palette": ["warm saturated primaries", "soft complementary accents", "clean color scripting"],
        "lighting": ["large soft key light", "warm rim light", "polished global illumination"],
        "composition": ["clear silhouette", "low camera for wonder", "character-centered framing"],
        "texture": ["soft plastic skin shading", "clean material definition", "highly groomed surfaces"],
        "negative_cues": ["photoreal horror", "muddy color", "flat sitcom lighting", "uncanny realism"],
    },
    "wes anderson": {
        "keywords": ["symmetrical framing", "deadpan staging", "miniature-like production design"],
        "palette": ["pastel pink", "mustard yellow", "powder blue", "muted red"],
        "lighting": ["soft frontal light", "even exposure", "storybook daylight"],
        "composition": ["centered subjects", "flat planes", "precise negative space"],
        "texture": ["paper props", "worn textiles", "handmade set dressing"],
        "negative_cues": ["handheld chaos", "gritty realism", "high contrast noir"],
    },
    "noir": {
        "keywords": ["film noir", "urban mystery", "dramatic silhouettes"],
        "palette": ["black", "silver", "deep blue", "smoke gray"],
        "lighting": ["hard side light", "venetian blind shadows", "low key contrast"],
        "composition": ["deep shadows", "oblique angles", "lonely figures"],
        "texture": ["rain slick streets", "cigarette smoke", "grain"],
        "negative_cues": ["bright sitcom lighting", "pastel whimsy", "flat daylight"],
    },
}


def analyze_style(request: MoodboardRequest) -> StyleBrief:
    directive = request.directive.strip()
    normalized = directive.lower()
    style_key = _match_known_style(normalized)
    base = STYLE_LIBRARY.get(style_key, _fallback_style(normalized))

    manual_context = ""
    if request.examples:
        manual_context = f" The user provided {len(request.examples)} manual example(s), so prioritize visual similarity to those seeds."

    keywords = _unique(base["keywords"] + _keywords_from_text(normalized))
    queries = _build_queries(directive, keywords, request.examples)

    return StyleBrief(
        directive=directive,
        summary=f"Collect references that express {directive} through specific, inspectable visual traits.{manual_context}",
        keywords=keywords[:10],
        palette=base["palette"],
        lighting=base["lighting"],
        composition=base["composition"],
        texture=base["texture"],
        negative_cues=base["negative_cues"],
        search_queries=queries,
    )


def _match_known_style(normalized: str) -> str | None:
    for key in STYLE_LIBRARY:
        if key in normalized:
            return key
    return None


def _fallback_style(normalized: str) -> dict[str, list[str]]:
    keywords = _keywords_from_text(normalized) or ["visual reference", "art direction", "style frame"]
    return {
        "keywords": keywords,
        "palette": ["dominant palette inferred from examples", "secondary accent colors", "contrast range"],
        "lighting": ["lighting style to infer", "shadow softness", "contrast pattern"],
        "composition": ["composition pattern", "subject scale", "camera distance"],
        "texture": ["surface treatment", "material language", "rendering detail"],
        "negative_cues": ["off-style references", "low quality images", "ambiguous matches"],
    }


def _keywords_from_text(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text)
    stop = {"the", "and", "for", "with", "style", "like", "into", "from", "this", "that"}
    return [word for word in words if len(word) > 2 and word not in stop][:8]


def _build_queries(directive: str, keywords: list[str], examples: list[str]) -> list[str]:
    core = " ".join(keywords[:4])
    queries = [
        f"{directive} reference board",
        f"{core} cinematic stills",
        f"{core} character design",
        f"{core} color script",
        f"{directive} environment design",
    ]
    if examples:
        queries.append(f"visually similar to provided examples {core}")
    return _unique(queries)


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return result

