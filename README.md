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

# Agentic Moodboard Workflow

This is an initial implementation scaffold for an agent that turns an art-style directive into a curated visual reference board.

The interview prompt:

> Given a style directive such as "Pixar style", optionally with manual examples, collect roughly 30-50 internet images in that style from sources like Pinterest and ShotDeck, then make the result usable for a director, prompt writer, or downstream Midjourney workflow.

This repo is intentionally built as a small, explainable prototype:

- A standard-library Python workflow that analyzes a style directive, proposes search/source strategies, scores candidate references, and emits a structured moodboard run.
- A lightweight live dashboard for launching a run and inspecting the agent's reasoning, candidates, scores, and next actions.
- Documentation that explains the architecture, tradeoffs, and path from mock prototype to production agent.

## Quick Start

Run the local dashboard:

```bash
python3 serve_dashboard.py
```

Then open:

```text
http://localhost:8787
```

Generate a sample run from the command line:

```bash
python3 run_demo.py --directive "Pixar style animated feature, warm family adventure, expressive characters" --count 36
```

The latest run is written to:

```text
data/sample_runs/latest.json
```

## Real Retrieval Gradio App

The mock dashboard above is useful for explaining orchestration. The model-backed version is `app.py`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Gradio app:

```bash
python3 app.py
```

This version does not train anything. It uses OpenCLIP for zero-shot image/text embeddings:

```text
directive + optional seed images
→ DuckDuckGo image candidates
→ image downloads
→ OpenCLIP text/image embeddings
→ cosine similarity ranking
→ near-duplicate removal
→ Gradio moodboard
```

On this Mac, the heavy dependencies may be slower. For the demo path, use the Colab instructions in [colab_quickstart.md](colab_quickstart.md).

The Gradio interface uses the public Hugging Face theme `Nymbo/Alyx_Theme`.

## Current Scope

There are now two paths:

- `serve_dashboard.py`: deterministic mock workflow for explaining architecture.
- `app.py`: real zero-shot retrieval workflow for producing actual image moodboards.

The real workflow uses web image search rather than Pinterest, ShotDeck, or Midjourney automation. Those should remain planned adapters until they are implemented with accounts, permissions, and browser automation.

The production version would replace `src/moodboard_agent/sources.py` with real adapters:

- Pinterest search or board collection
- ShotDeck search
- General web image search
- Manual example image analysis
- Optional Midjourney upload/browser automation

## Project Layout

```text
.
├── config/
│   └── pipeline_config.json
├── dashboard/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── data/
│   └── sample_runs/
├── docs/
│   ├── architecture.md
│   ├── explain-tomorrow.md
│   └── roadmap.md
├── run_demo.py
├── serve_dashboard.py
└── src/
    └── moodboard_agent/
        ├── __init__.py
        ├── pipeline.py
        ├── schemas.py
        ├── scoring.py
        ├── server.py
        ├── sources.py
        ├── storage.py
        └── style_analysis.py
```

## What To Emphasize In The Interview

The core idea is not "I built a scraper." The stronger framing is:

1. I decomposed the agentic workflow into clear stages.
2. I made each stage observable and debuggable through a dashboard.
3. I defined data contracts so real source adapters can replace mocks without changing the whole app.
4. I separated subjective creative judgment into explicit scoring dimensions that a director can inspect and override.
5. I left room for human-in-the-loop approval before fragile or account-sensitive steps like Pinterest/ShotDeck browsing or Midjourney uploads.
