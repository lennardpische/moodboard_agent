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

# Moodboard Agent 🎬

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue)

**Live demo:** [lpiske-moodboard-workflow.hf.space](https://lpiske-moodboard-workflow.hf.space)

---

## The Problem

Text descriptions of visual style are fundamentally ambiguous — *"warm lighting"* means something different to every model and every director. Creative directors spend hours assembling moodboards manually to communicate precise visual intent. This tool automates that process: give it one sentence describing a style and it retrieves, ranks, and deduplicates 30 on-brief reference images, then generates a new image in that style and exports a Midjourney `--sref` command using the actual moodboard images.

---

## The Tech

- **OpenCLIP ViT-B-32 / LAION-2B** — embeds images and text into the same 512-dim vector space for zero-shot cosine similarity scoring
- **DuckDuckGo image search** — candidate retrieval with threaded 8s timeouts to handle DDG's unofficial API hanging
- **Claude Haiku** — structured style brief generation for unknown styles (~$0.0005/call)
- **FLUX.1-schnell** — image generation from brief-derived prompt (not the raw directive)
- **Gradio** — UI with live gallery, ranked results table, and Midjourney export

---

## How It Works

```
style directive
  → style analysis (library / Claude Haiku / keyword fallback)
  → search query generation + exploration variation
  → DuckDuckGo image search (threaded, 8s timeout per query)
  → download + 4-layer quality filtering
  → OpenCLIP ViT-B-32 embedding (text + images + negative cues)
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

**Optional credentials:**

| Variable | What it enables |
|---|---|
| `ANTHROPIC_API_KEY` | Claude Haiku style analysis for unknown styles |
| `HF_TOKEN` | FLUX.1-schnell image generation |

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export HF_TOKEN=hf_...
python3 app.py
```

Without credentials the app still runs — known styles (Pixar, Wes Anderson, noir) use the built-in library and the moodboard works fully. Generation requires `HF_TOKEN`.

---

## Key Design Decisions

**Negative embedding for content filtering** — The style brief includes `negative_cues` (e.g. for Pixar: *"photoreal horror, uncanny realism"*). These are embedded and their similarity is subtracted from each candidate's score (`score − 0.3 × negative_similarity`), penalising photorealistic content in animation-style queries without a hard content filter.

**Seeded exploration** — Without randomness the pipeline is fully deterministic — same prompt, same 30 images every run. The exploration slider drives weighted sampling (`score ^ (1/exploration)`) from the top-ranked pool and adds query suffix variants. Every run's seed is shown in the status bar for reproducibility.

**ThreadPoolExecutor timeout** — DuckDuckGo's unofficial API hangs indefinitely on rate-limited queries. `signal.alarm` doesn't work in Gradio's worker threads. Each query runs in an isolated thread with an 8s deadline via `future.result(timeout=8)`.

**FLUX prompt from brief attributes** — The generation prompt uses extracted keywords, palette, and lighting descriptors, not the raw directive. Diffusion models respond better to specific visual vocabulary than human-readable style descriptions.

---

## Controls

| Control | What it does |
|---|---|
| **Exploration** | 0 = deterministic. Higher = query variation + weighted sampling from top pool. |
| **Text weight** | Balance between style brief and uploaded seed images when scoring. |
| **Dedupe threshold** | Cosine similarity cutoff for near-duplicate removal (default 0.92). |
| **Seed images** | Upload references — embeddings averaged and blended with text score. |

---

## Project Layout

```
app.py                          — Gradio UI, event wiring, FLUX generation
requirements.txt
src/moodboard_agent/
  style_analysis.py             — style brief generation (library / LLM / fallback)
  live_retrieval.py             — full pipeline (search → embed → rank → select)
  schemas.py                    — shared dataclasses
  pipeline.py                   — mock pipeline (architecture reference)
  sources.py                    — mock source adapters (Pinterest/ShotDeck stubs)
tests/
  test_live_retrieval.py        — unit tests (exploration, diversity, query variation)
  test_pipeline.py              — unit tests for mock pipeline
```

---

## Next Steps

- Pinterest / ShotDeck adapters for curated sources over broad web search
- Human review loop — pin/reject UI that re-ranks candidates via embedding feedback
- Stable image hosting (S3/R2) for permanent Midjourney `--sref` URLs
- IP-Adapter generation — true image-conditioned output (moodboard pixels in, not text)
- Fine-tuned CLIP on (directive, approved image) pairs from real director sessions

---

## License

MIT
