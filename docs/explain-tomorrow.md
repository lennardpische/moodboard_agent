# How To Explain This Tomorrow

## Thirty-Second Version

I built a prototype for an agentic moodboard workflow. The user gives it a style directive like "Pixar style", optionally with seed examples. The system turns that into a structured style brief, searches for candidate images, embeds the text and images with OpenCLIP, ranks by visual/text similarity, removes near-duplicates, and shows the result in a Gradio dashboard.

I kept a mock architecture dashboard too, but the Gradio path is the real model-backed version. It does not train a new model; it uses a pretrained vision-language model for zero-shot retrieval.

## The Key Point

The value is not just grabbing images. The value is making a loose creative phrase operational:

- What visual traits does "Pixar style" actually imply?
- Which searches should be run?
- Which images are useful references rather than random matches?
- Where should a human director approve or correct the agent?
- How does the result become usable by a prompting agent or Midjourney workflow?

## Walkthrough

1. Start at the dashboard.
2. Enter a directive.
3. Add one or two optional examples.
4. Run the workflow.
5. Show the style brief.
6. Show source plans and candidate scoring.
7. Show selected references and next actions.

## Design Decisions

Pretrained embeddings first:

I did not fine-tune because I do not have labeled studio preference data yet. A pretrained model like OpenCLIP is the right first step because it already maps images and text into the same embedding space.

Mock adapters still exist:

This avoids pretending Pinterest, ShotDeck, or Midjourney automation is already solved. Those are future adapters once account access, permissions, and reliability are handled.

Structured JSON everywhere:

Every stage emits JSON. That makes the system inspectable, testable, and easy to connect to a dashboard or another agent.

Scoring is explicit:

The model separates style match, visual variety, composition value, prompting utility, and rights safety. Those dimensions are subjective, but exposing them makes the agent's judgment reviewable.

Human review remains central:

The agent should recommend and organize. A director or producer should be able to pin, reject, annotate, or steer the board.

## What I Would Build Next

1. Better source adapters with stable APIs or Playwright where permitted.
2. Stronger captioning/explanation using BLIP or Florence-2.
3. Review controls: accept, reject, pin, more like this, less like this.
4. A review queue with accept, reject, pin, and replace controls.
5. Midjourney handoff: either browser automation or a generated upload package.

## Risks And How I Would Handle Them

Scraping reliability:

Use adapters with small, testable responsibilities. Keep fallback sources. Cache results.

Copyright and usage rights:

Store source URLs and rights notes. Use the moodboard internally as reference material, not as final output.

Style ambiguity:

Ask for manual examples when the directive is vague. Show the extracted style brief so the user can correct it.

Near-duplicate images:

Use embedding or perceptual-hash dedupe before final selection.

Too much automation:

Keep review checkpoints before external account actions, especially Midjourney uploads.

## Strong Interview Framing

I treated this as an orchestration problem first. The agent needs tools, memory, state, scoring, source adapters, and a UI for human feedback. Once that spine is clear, adding stronger models or real browser workers is straightforward.
