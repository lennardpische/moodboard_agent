---
title: Moodboard Retrieval Agent
emoji: 🎬
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.14.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
---

# Moodboard Agent

Zero-shot visual retrieval pipeline that turns a plain-English style directive into a curated 30-image moodboard, with FLUX image generation and Midjourney `--sref` export.

**Live demo:** [lpiske-moodboard-workflow.hf.space](https://lpiske-moodboard-workflow.hf.space)

---

## What it does

You type a style directive — *"Pixar style animated feature, warm family adventure, expressive characters"* — and the pipeline:

1. Analyses the style into specific visual attributes (palette, lighting, composition, texture) using a built-in library for known styles or Claude Haiku for unknown ones
2. Builds DuckDuckGo image search queries from those attributes
3. Downloads and filters candidates (HTTP status, Content-Type, PIL verification, minimum dimensions)
4. Embeds every image and the style brief with OpenCLIP, scores by cosine similarity, and penalises off-style content using the brief's negative cues
5. Runs a greedy deduplication pass to ensure visual variety across the 30 selected images
6. Generates a new image with FLUX.1-schnell using a prompt built from the brief's extracted attributes — not your original directive verbatim
7. Exports a Midjourney `/imagine --sref` command using the actual moodboard images as style references

---

## Pipeline

```
style directive
  → style analysis (library / Claude Haiku / keyword fallback)
  → search query generation + variation
  → DuckDuckGo image search (threaded, 8s timeout per query)
  → download + 4-layer quality filtering
  → OpenCLIP ViT-B-32 embedding (text + images)
  → cosine similarity ranking with negative cue penalty
  → greedy diversity deduplication (cosine threshold 0.92)
  → 30-image moodboard
  → FLUX.1-schnell generation (brief-derived prompt)
  → Midjourney --sref export
```

---

## Setup

```bash
pip install -r requirements.txt
python3 app.py
```

Opens at `http://127.0.0.1:7860`.

**Optional credentials** (set as environment variables or HF Spaces secrets):

| Variable | What it enables |
|---|---|
| `ANTHROPIC_API_KEY` | Claude Haiku style analysis for unknown styles (~$0.0005/call) |
| `HF_TOKEN` | FLUX.1-schnell image generation |

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export HF_TOKEN=hf_...
python3 app.py
```

Without these, the app still runs: known styles (Pixar, Wes Anderson, noir) use the built-in library, and the generation section requires HF_TOKEN.

---

## Controls

| Control | What it does |
|---|---|
| **Text weight** | Balance between style brief (text) and uploaded seed images when scoring candidates. Only active if you upload seeds. |
| **Exploration** | 0 = fully deterministic (same seed = same moodboard). Higher values add query variation and weighted sampling from the top-ranked pool. |
| **Dedupe threshold** | Cosine similarity cutoff for near-duplicate removal. 0.92 removes near-identical images while keeping genuine variety. |
| **Seed images** | Upload reference images — their embeddings are averaged and blended with the text score at the ratio set by text weight. |

---

## Key technical decisions

**OpenCLIP over supervised ViT** — ViT-B-32 trained on LAION-2B embeds images and text into the same 512-dim space. This allows direct cosine similarity scoring between a style description and candidate images with no task-specific training. LAION-2B's large proportion of artistic and cinematic content gives strong zero-shot performance on style-based queries.

**Negative embedding** — The style brief includes `negative_cues` (e.g. for Pixar: `"photoreal horror, uncanny realism"`). Those cues are embedded and their similarity is subtracted from each candidate's score (`score − 0.3 × negative_similarity`), penalising photorealistic content in animation-style queries.

**ThreadPoolExecutor timeout for DDG** — DuckDuckGo's unofficial API occasionally hangs indefinitely on rate-limited queries. Each query runs in an isolated thread with an 8-second deadline. `signal.alarm` doesn't work because Gradio runs callbacks in worker threads, not the main thread.

**Seeded exploration** — The exploration slider drives weighted sampling from the top-ranked candidate pool (`score ^ (1/exploration)`) and query suffix variation. Every run's seed is displayed so any result can be reproduced exactly.

**FLUX prompt from brief attributes** — The generation prompt is built from extracted keywords, palette, and lighting descriptors, not from the original directive. Diffusion models respond better to specific visual vocabulary than to human-readable style descriptions.

---

## Project layout

```
app.py                          — Gradio UI, event wiring, generation
requirements.txt
src/moodboard_agent/
  style_analysis.py             — style brief generation (library / LLM / fallback)
  live_retrieval.py             — full pipeline (search → embed → rank → select)
  schemas.py                    — shared dataclasses
  pipeline.py                   — mock pipeline (architecture reference)
  sources.py                    — mock source adapters (Pinterest/ShotDeck stubs)
  scoring.py                    — mock scoring (architecture reference)
tests/
  test_live_retrieval.py        — unit tests (exploration, diversity selection, query variation)
  test_pipeline.py              — unit tests for mock pipeline
```

---

## What a production version would add

1. **Pinterest / ShotDeck adapters** — curated sources over broad web search
2. **Human review loop** — pin/reject UI that re-ranks remaining candidates via embedding feedback
3. **Stable image hosting** — S3/R2 so Midjourney `--sref` URLs are permanent
4. **IP-Adapter generation** — true image-conditioned generation (moodboard pixels in, not text)
5. **Fine-tuned CLIP** — trained on (directive, approved image) pairs from real director sessions

---

## License

MIT
