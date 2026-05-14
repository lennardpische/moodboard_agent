#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from moodboard_agent.live_retrieval import RetrievalConfig, run_live_retrieval


DEFAULT_PROMPT = "Pixar style animated feature, warm family adventure, expressive characters, soft lighting, colorful environments"


def build_dashboard() -> gr.Blocks:
    with gr.Blocks(title="Moodboard Agent") as demo:
        gr.Markdown(
            """
            # Moodboard Retrieval Agent

            Zero-shot visual retrieval using web image candidates and OpenCLIP image/text embeddings.
            """
        )

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
                    target_count = gr.Slider(8, 50, value=12, step=1, label="Images in moodboard")
                    candidate_count = gr.Slider(20, 240, value=40, step=10, label="Candidates to retrieve")
                with gr.Row():
                    text_weight = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Text weight")
                    diversity_threshold = gr.Slider(0.75, 0.99, value=0.92, step=0.01, label="Dedupe threshold")
                run_button = gr.Button("Build moodboard", variant="primary")

            with gr.Column(scale=1):
                brief = gr.Textbox(label="Style brief", lines=8)
                queries = gr.Textbox(label="Search queries", lines=8)
                status = gr.Textbox(label="Run status", lines=4)

        gallery = gr.Gallery(
            label="Selected references",
            columns=[2, 3, 4],
            rows=[2, 3, 4],
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

        run_button.click(
            fn=run_dashboard,
            inputs=[directive, seed_images, target_count, candidate_count, text_weight, diversity_threshold],
            outputs=[brief, queries, status, gallery, table],
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
    )


def run_dashboard(
    directive: str,
    seed_images: list[object] | None,
    target_count: int,
    candidate_count: int,
    text_weight: float,
    diversity_threshold: float,
) -> tuple[str, str, str, list[tuple[object, str]], list[list[object]]]:
    seed_paths = [str(getattr(item, "name", item)) for item in seed_images or []]
    config = RetrievalConfig(
        target_count=int(target_count),
        candidate_count=int(candidate_count),
        text_weight=float(text_weight),
        diversity_threshold=float(diversity_threshold),
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
        [
            idx + 1,
            round(item.score, 4),
            item.title,
            item.source,
            item.image_url,
            item.page_url,
        ]
        for idx, item in enumerate(result.selected)
    ]

    status = (
        f"Collected {result.downloaded_count} usable image(s) from {result.raw_candidate_count} raw candidate(s).\n"
        f"Selected {len(result.selected)} image(s).\n"
        f"Model: {result.model_name}"
    )
    return result.brief.summary, "\n".join(result.brief.search_queries), status, gallery_items, rows


if __name__ == "__main__":
    launch_dashboard()
