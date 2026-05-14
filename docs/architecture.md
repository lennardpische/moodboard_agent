# Architecture

## Goal

Build an agentic workflow that accepts an art-style directive and produces a director-ready moodboard of 30-50 reference images. The workflow should support a live dashboard now and later connect to real web collection, visual analysis, and Midjourney submission.

## Agent Loop

The prototype uses a six-stage loop:

1. Intake
   - Input: style directive, optional manual example URLs or notes, target image count.
   - Output: normalized task request.

2. Style Analysis
   - Converts loose creative language into a structured style brief.
   - Extracts visual cues such as subject, palette, lighting, texture, composition, and negative constraints.

3. Source Planning
   - Chooses where to search and what queries to run.
   - Current adapters are mocks for Pinterest, ShotDeck, web image search, and manual example expansion.

4. Candidate Collection
   - Gathers candidate image records.
   - Production adapters would capture thumbnail URL, source URL, source name, metadata, and collection notes.

5. Scoring And Selection
   - Scores each candidate on style match, variety, composition value, prompting utility, and rights safety.
   - Selects a diverse final set rather than simply taking the top N near-duplicates.

6. Review And Export
   - Exposes results in the dashboard.
   - Future exports: CSV, contact sheet, Midjourney upload queue, director review board, or prompt context package.

## Why Mock First

Pinterest, ShotDeck, and Midjourney all have practical constraints: accounts, login state, rate limits, UI changes, and rights considerations. A mock-first structure lets the workflow be demonstrated and reasoned about before adding brittle automation.

The important engineering decision is that `sources.py` returns the same `CandidateImage` records that real adapters will return later. That makes the dashboard, scoring, storage, and export code stable while source collection improves.

## Real Retrieval Prototype

The Gradio app in `app.py` adds a model-backed path:

1. Generate search queries from the directive.
2. Retrieve web image candidates through DuckDuckGo image search.
3. Download usable images.
4. Embed the style brief as text with OpenCLIP.
5. Embed each candidate image with OpenCLIP.
6. Optionally embed uploaded seed images.
7. Rank candidates by cosine similarity.
8. Remove near-duplicates using image-embedding similarity.
9. Display selected references, scores, and source URLs in Gradio.

This is zero-shot retrieval. It does not require a custom dataset or reward-model training.

## Data Model

The main objects are:

- `MoodboardRequest`: directive, examples, target count.
- `StyleBrief`: extracted visual language and search terms.
- `CandidateImage`: one possible image reference with source metadata and scores.
- `MoodboardRun`: full trace of a run, including plan, selected images, rejected images, and next actions.

Every object serializes to JSON so it can be inspected, saved, served to the dashboard, or passed to another agent.

## Production Upgrade Path

The production version should add:

- Playwright browser workers for Pinterest, ShotDeck, and Midjourney interactions.
- A search API or browser search adapter for general web images.
- Image embedding similarity to compare candidates against manual examples.
- Vision-model analysis of seed artwork and collected candidates.
- Deduplication by perceptual hash or embedding similarity.
- Human review controls for accepting, rejecting, pinning, and annotating references.
- Export adapters for Midjourney, Figma, Notion, PDF contact sheets, or an internal production database.

## Human-In-The-Loop Control

Creative automation should not hide judgment. The dashboard should show:

- Why the agent searched a given query.
- Why an image was selected or rejected.
- Which sources produced the strongest references.
- Which references are uncertain or need manual review.
- What the agent would do next if allowed to continue.

That keeps the agent useful without pretending that style judgment is fully objective.
