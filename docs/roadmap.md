# Roadmap

## Phase 1: Working Skeleton

Status: implemented in this scaffold.

- Accept a style directive and optional examples.
- Generate a structured style brief.
- Generate a source plan.
- Produce mock candidate images.
- Score and select references.
- Save a run as JSON.
- View the run in a live dashboard.

## Phase 2: Real Collection

- Add Playwright.
- Create source adapter interface tests.
- Implement Pinterest search adapter.
- Implement ShotDeck search adapter.
- Implement general web image adapter.
- Capture screenshot evidence for each adapter.
- Cache raw source results.

## Phase 3: Visual Intelligence

- Add image download/cache layer.
- Analyze manual examples with a vision model.
- Compute image embeddings.
- Deduplicate candidates.
- Cluster references by composition, palette, character design, lighting, and environment.
- Add "replace this image with more like it" behavior.

## Phase 4: Production Dashboard

- Add run history.
- Add accept, reject, pin, annotate, and regenerate controls.
- Add board sections such as characters, environments, lighting, materials, and camera language.
- Add export to CSV, PDF contact sheet, Notion, Figma, or internal tools.

## Phase 5: Midjourney Handoff

- Generate a Midjourney upload queue.
- Add human approval before uploads.
- Automate browser submission where permitted.
- Track which references were submitted and which prompts they supported.

