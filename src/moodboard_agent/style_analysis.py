from __future__ import annotations

import json
import os
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

    manual_context = ""
    if request.examples:
        manual_context = f" The user provided {len(request.examples)} manual example(s), so prioritize visual similarity to those seeds."

    if style_key:
        base = STYLE_LIBRARY[style_key]
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

    # Unknown style — try LLM, fall back to keyword extraction
    llm, llm_status = _llm_analyze_style(directive)
    if llm:
        queries = _unique(llm.get("search_queries") or _build_queries(directive, llm.get("keywords", []), request.examples))
        return StyleBrief(
            directive=directive,
            summary=f"Collect references that express {directive} through specific, inspectable visual traits.{manual_context}",
            keywords=_unique(llm.get("keywords", []))[:10],
            palette=llm.get("palette", []),
            lighting=llm.get("lighting", []),
            composition=llm.get("composition", []),
            texture=llm.get("texture", []),
            negative_cues=llm.get("negative_cues", []),
            search_queries=queries[:8],
            llm_used=True,
            llm_error="",
        )
    # LLM failed — store the reason so the UI can show it
    base = _fallback_style(normalized)
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
        llm_used=False,
        llm_error=llm_status if llm_status != "no API key" else "",
    )



def _llm_analyze_style(directive: str) -> tuple[dict | None, str]:
    """Returns (result_dict_or_None, status_message)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None, "no API key"
    try:
        import anthropic
    except ImportError:
        return None, "anthropic package not installed"
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": (
                    "You are a visual development consultant. "
                    "Given the art-style directive below, return a JSON object with these exact fields:\n"
                    "- keywords: list of 6-10 specific visual style keywords\n"
                    "- palette: list of 3-5 color description strings\n"
                    "- lighting: list of 2-4 lighting description strings\n"
                    "- composition: list of 2-4 composition description strings\n"
                    "- texture: list of 2-4 texture/material description strings\n"
                    "- negative_cues: list of 3-5 visual things to avoid\n"
                    "- search_queries: list of 5-7 image search query strings for finding reference images\n\n"
                    f"Directive: {directive}\n\n"
                    "Return ONLY valid JSON. No explanation, no markdown fences."
                ),
            }],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw), "ok"
    except Exception as exc:
        return None, str(exc)


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
