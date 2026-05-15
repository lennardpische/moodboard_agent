#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from moodboard_agent.live_retrieval import RetrievalConfig, run_live_retrieval


DEFAULT_PROMPT = "Pixar style animated feature, warm family adventure, expressive characters, soft lighting, colorful environments"

_MJ_PLACEHOLDER = "Build a moodboard first, then type your scene above."

# ── A: stable URL resolution ──────────────────────────────────────────────────
# On HF Spaces, SPACE_HOST is set (e.g. "owner-spacename.hf.space").
# Gradio serves any file that was returned through its outputs at /file=<path>.
# That URL is publicly reachable, so Midjourney can fetch it.
# Locally, 127.0.0.1:7860 is not reachable by Midjourney's servers, so we fall
# back to the original source URL from DuckDuckGo (may still fail on hotlinked CDNs).

def _public_url(local_path: str, fallback_url: str) -> str:
    space_host = os.environ.get("SPACE_HOST", "")
    if space_host:
        return f"https://{space_host}/file={local_path}"
    return fallback_url


def _url_note() -> str:
    if os.environ.get("SPACE_HOST"):
        return "URLs: served via this Space — stable for Midjourney."
    return "URLs: using original source URLs (local server not reachable by Midjourney — deploy to HF Spaces for stable links)."


# ── C: LLM indicator ─────────────────────────────────────────────────────────

def _llm_status(llm_used: bool, llm_error: str = "") -> str:
    if llm_used:
        return "LLM style analysis: used (Claude Haiku generated the brief)"
    if llm_error:
        return f"LLM style analysis: failed — {llm_error}"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "LLM style analysis: available but not needed (known style matched)"
    return "LLM style analysis: OFF (set ANTHROPIC_API_KEY to enable for unknown styles)"


# ── Image-conditioned generation via HF Inference API ────────────────────────

def _hf_token() -> str:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    )


def _scene_text(mj_prompt: str) -> str:
    """Extract only the scene/subject text — strip style keywords and all MJ flags."""
    import re
    text = re.sub(r"^/imagine prompt:\s*", "", mj_prompt)
    text = re.split(r"\s+--\w+", text)[0]
    # Strip the style keywords appended after the first comma group if they look like brief keywords
    return text.strip()


def _make_style_collage(image_paths: list[str]):
    """Tile the top moodboard images into a 2×2 grid as a single style reference."""
    try:
        from PIL import Image as PILImage
        import math
        size = 512
        cols = 2
        paths = [p for p in image_paths[:4] if Path(p).exists()]
        if not paths:
            return None
        imgs = []
        for p in paths:
            try:
                imgs.append(PILImage.open(p).convert("RGB").resize((size, size)))
            except Exception:
                continue
        if not imgs:
            return None
        rows = math.ceil(len(imgs) / cols)
        collage = PILImage.new("RGB", (size * min(len(imgs), cols), size * rows), (20, 20, 20))
        for i, img in enumerate(imgs):
            col, row = i % cols, i // cols
            collage.paste(img, (col * size, row * size))
        return collage
    except Exception:
        return None


def _build_generation_prompt(scene: str, state: dict) -> str:
    """Build a FLUX prompt from the moodboard brief — never copies the raw directive."""
    keywords = state.get("keywords", [])
    palette = state.get("palette", [])
    lighting = state.get("lighting", [])
    parts = [scene.strip()] if scene.strip() else []
    if keywords:
        parts.append(", ".join(keywords[:5]))
    if palette:
        parts.append(", ".join(palette[:2]))
    if lighting:
        parts.append(lighting[0])
    return ", ".join(parts)


def generate_preview(mj_prompt: str, state: dict):
    images = state.get("images", [])
    if not images:
        return None, None, "Build a moodboard first."

    token = _hf_token()
    if not token:
        return None, None, (
            "HF_TOKEN not set.\n"
            "Get one at huggingface.co → Settings → Access Tokens, "
            "then: export HF_TOKEN=hf_... (locally) or add it as a Space secret."
        )

    scene = _scene_text(mj_prompt)
    if not scene or scene == _MJ_PLACEHOLDER:
        return None, None, "Type a scene above first."

    top_paths = [img["local_path"] for img in images[:4]]
    collage = _make_style_collage(top_paths)
    if collage is None:
        return None, None, "Could not load moodboard images for style reference."

    flux_prompt = _build_generation_prompt(scene, state)
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token)
        result = client.text_to_image(
            prompt=flux_prompt,
            model="black-forest-labs/FLUX.1-schnell",
        )
        note = (
            f"Prompt fed to FLUX: {flux_prompt}\n\n"
            "Style terms (keywords, palette, lighting) were extracted from the moodboard brief — "
            "not copied from your directive. The collage shows the top 4 images whose analysis shaped the prompt."
        )
        return collage, result, note
    except Exception as exc:
        return collage, None, f"Generation failed: {exc}"


# ── Midjourney prompt builder ─────────────────────────────────────────────────

def build_mj_prompt(
    scene_text: str,
    sref_count: int,
    style_weight: int,
    aspect_ratio: str,
    append_keywords: bool,
    selected_paths: list[str],
    state: dict,
) -> str:
    images = state.get("images", [])
    directive = state.get("directive", "")
    keywords = state.get("keywords", [])

    if not images:
        return _MJ_PLACEHOLDER

    # Build a lookup from local_path → image info
    by_path = {img["local_path"]: img for img in images}

    # Filter to what the user checked, preserve score order, cap at sref_count
    ordered = [img for img in images if img["local_path"] in set(selected_paths)]
    sref_images = ordered[:int(sref_count)]

    text_parts = []
    if scene_text.strip():
        text_parts.append(scene_text.strip())
    elif directive:
        text_parts.append(directive)

    if append_keywords and keywords:
        text_parts.append(", ".join(keywords[:6]))

    prompt_text = ", ".join(text_parts) if text_parts else "cinematic scene"

    cmd = f"/imagine prompt: {prompt_text}"

    if sref_images:
        urls = [_public_url(img["local_path"], img["image_url"]) for img in sref_images]
        cmd += f" --sref {' '.join(urls)}"

    if int(style_weight) != 100:
        cmd += f" --sw {int(style_weight)}"

    cmd += f" --ar {aspect_ratio}"
    return cmd


# ── Dashboard ─────────────────────────────────────────────────────────────────

def build_dashboard() -> gr.Blocks:
    with gr.Blocks(title="Moodboard Agent") as demo:
        gr.Markdown(
            """
            # Moodboard Retrieval Agent

            Zero-shot visual retrieval using web image candidates and OpenCLIP image/text embeddings.
            """
        )

        mj_state = gr.State({"images": [], "directive": "", "keywords": []})

        with gr.Row():
            with gr.Column(scale=1):
                directive = gr.Textbox(
                    label="Style directive",
                    value=DEFAULT_PROMPT,
                    lines=5,
                )
                seed_images = gr.Files(
                    label="Optional seed images",
                    file_types=["image"],
                    file_count="multiple",
                )
                with gr.Row():
                    target_count = gr.Slider(8, 50, value=30, step=1, label="Images in moodboard")
                    candidate_count = gr.Slider(20, 240, value=100, step=10, label="Candidates to retrieve")
                with gr.Row():
                    text_weight = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Text weight")
                    diversity_threshold = gr.Slider(0.75, 0.99, value=0.92, step=0.01, label="Dedupe threshold")
                exploration = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="Exploration (0 = deterministic, 1 = max variety)")
                run_button = gr.Button("Build moodboard", variant="primary")

            with gr.Column(scale=1):
                brief = gr.Textbox(label="Style brief", lines=8)
                queries = gr.Textbox(label="Search queries", lines=8)
                status = gr.Textbox(label="Run status", lines=5)

        gallery = gr.Gallery(
            label="Moodboard",
            columns=5,
            height="auto",
            show_label=True,
            object_fit="cover",
        )
        table = gr.Dataframe(
            label="Ranked results",
            headers=["rank", "score", "title", "source", "image_url", "page_url"],
            datatype=["number", "number", "str", "str", "str", "str"],
            wrap=True,
        )

        # ── Generation section ────────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown(
            """
            ## Generate

            Type a scene — the **subject** you want to create in the moodboard's style.
            Example: moodboard = Pixar style → scene = *"a young girl discovering a hidden door in a library"*.
            The moodboard brief (keywords, palette, lighting) shapes the FLUX prompt — your directive is not reused verbatim.
            """
        )

        mj_scene = gr.Textbox(
            label="Scene / subject",
            placeholder="e.g. a young girl discovering a hidden door in a library",
            lines=2,
        )

        # B: interactive image selection
        mj_image_select = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="Style reference images — uncheck any you want to exclude",
        )

        with gr.Row():
            mj_sref_count = gr.Slider(1, 5, value=3, step=1, label="Max style references")
            mj_aspect_ratio = gr.Dropdown(
                choices=["16:9", "4:3", "1:1", "3:2", "9:16", "3:4", "2:3"],
                value="16:9",
                label="Aspect ratio",
            )

        mj_append_keywords = gr.Checkbox(
            value=True,
            label="Append style keywords from brief",
        )

        preview_button = gr.Button("Generate with FLUX (brief-guided)", variant="primary")

        with gr.Row():
            collage_image = gr.Image(label="Top 4 moodboard images (style extracted from these)", show_label=True)
            preview_image = gr.Image(label="Generated image", show_label=True)
        preview_note = gr.Textbox(label="Generation info", lines=4, interactive=False)

        # Midjourney (secondary — copy-paste workflow)
        gr.Markdown(
            "**Midjourney** — if you have a subscription, copy this command into Midjourney. "
            "`--sref` feeds the actual moodboard images into Midjourney's vision encoder for stronger style fidelity."
        )

        mj_style_weight = gr.Slider(0, 1000, value=100, step=10, label="Midjourney style weight (--sw)")

        mj_prompt_output = gr.Textbox(
            label="/imagine prompt — copy and paste into Midjourney",
            value=_MJ_PLACEHOLDER,
            lines=4,
            interactive=False,
        )

        mj_inputs = [mj_scene, mj_sref_count, mj_style_weight, mj_aspect_ratio, mj_append_keywords, mj_image_select, mj_state]

        # ── Event wiring ──────────────────────────────────────────────────────
        run_button.click(
            fn=run_dashboard,
            inputs=[directive, seed_images, target_count, candidate_count, text_weight, diversity_threshold, exploration],
            outputs=[brief, queries, status, gallery, table, mj_state, mj_image_select, mj_prompt_output],
        )

        for component in [mj_scene, mj_sref_count, mj_style_weight, mj_aspect_ratio, mj_append_keywords, mj_image_select]:
            component.change(fn=build_mj_prompt, inputs=mj_inputs, outputs=mj_prompt_output)

        preview_button.click(
            fn=generate_preview,
            inputs=[mj_prompt_output, mj_state],
            outputs=[collage_image, preview_image, preview_note],
        )

        gr.Examples(
            examples=[
                ["Pixar style animated feature, warm family adventure, expressive characters, soft lighting, colorful environments"],
                ["Wes Anderson inspired hotel interior, symmetrical framing, pastel palette, deadpan composition"],
                ["Film noir detective story, rainy city streets, hard shadows, venetian blind lighting, high contrast"],
                ["Solarpunk city marketplace, optimistic future, greenery integrated with architecture, bright natural light"],
                ["Dark fantasy forest shrine, ancient stone, moss, candlelight, mist, ritual atmosphere"],
                ["1990s Japanese cyberpunk anime city, neon signs, rainy streets, dense urban detail"],
                ["Luxury perfume campaign, macro florals, glass reflections, soft gold lighting, editorial composition"],
                ["Claymation children's storybook, handmade textures, cozy village, soft colors, tactile materials"],
            ],
            inputs=[directive],
        )

    return demo


def app_theme() -> gr.Theme:
    return gr.themes.Soft()


def launch_dashboard() -> None:
    server_name = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    build_dashboard().launch(
        server_name=server_name,
        server_port=7860,
        theme=app_theme(),
        allowed_paths=["/tmp"],
    )


def run_dashboard(
    directive: str,
    seed_images: list[object] | None,
    target_count: int,
    candidate_count: int,
    text_weight: float,
    diversity_threshold: float,
    exploration: float = 0.25,
) -> tuple:
    seed_paths = [str(getattr(item, "name", item)) for item in seed_images or []]
    config = RetrievalConfig(
        target_count=int(target_count),
        candidate_count=int(candidate_count),
        text_weight=float(text_weight),
        diversity_threshold=float(diversity_threshold),
        exploration=float(exploration),
    )
    result = run_live_retrieval(directive, seed_image_paths=seed_paths, config=config)

    gallery_items = [
        (
            item.local_path,
            f"{idx + 1}. {item.title}\nscore={item.score:.3f} | {item.source}",
        )
        for idx, item in enumerate(result.selected)
    ]
    rows = [
        [idx + 1, round(item.score, 4), item.title, item.source, item.image_url, item.page_url]
        for idx, item in enumerate(result.selected)
    ]

    llm_line = _llm_status(result.brief.llm_used, result.brief.llm_error)
    url_line = _url_note()
    status = (
        f"Collected {result.downloaded_count} usable image(s) from {result.raw_candidate_count} raw candidate(s).\n"
        f"Selected {len(result.selected)} image(s).\n"
        f"Model: {result.model_name}\n"
        f"Exploration: {result.exploration:.2f} | Seed: {result.used_seed}\n"
        f"{llm_line}"
    )

    # B: build checkbox choices — (display label, local_path) tuples
    images_meta = [
        {
            "local_path": item.local_path,
            "image_url": item.image_url,
            "title": item.title,
            "score": item.score,
        }
        for item in result.selected
    ]
    checkbox_choices = [
        (f"{idx + 1}. {item.title[:48]}  ({item.score:.3f})", item.local_path)
        for idx, item in enumerate(result.selected)
    ]
    all_paths = [item.local_path for item in result.selected]

    state = {
        "images": images_meta,
        "directive": directive,
        "keywords": result.brief.keywords[:8],
        "palette": result.brief.palette,
        "lighting": result.brief.lighting,
        "url_note": url_line,
    }

    initial_mj = build_mj_prompt("", 3, 100, "16:9", True, all_paths, state)
    checkbox_update = gr.update(choices=checkbox_choices, value=all_paths)

    return (
        result.brief.summary,
        "\n".join(result.brief.search_queries),
        status,
        gallery_items,
        rows,
        state,
        checkbox_update,
        initial_mj,
    )


if __name__ == "__main__":
    launch_dashboard()
