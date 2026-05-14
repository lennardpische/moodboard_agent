# Hugging Face Spaces Deploy

This repo is ready to run as a Gradio Space.

## Fastest Path

1. Create a Hugging Face account or log in.
2. Go to `https://huggingface.co/spaces`.
3. Click `Create new Space`.
4. Use:
   - Space SDK: `Gradio`
   - Visibility: public or private
   - Hardware: CPU basic first, GPU if CPU is too slow
5. Upload or push this repo with these files at the Space root:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `src/`
   - `docs/`
   - `config/`

Hugging Face will install `requirements.txt` and run `app.py`.

## Git Push Path

Install Git LFS if Hugging Face asks for it:

```bash
brew install git-lfs
git lfs install
```

Clone the Space repo:

```bash
git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
```

Copy this project into the cloned Space folder, then:

```bash
git add .
git commit -m "Add moodboard retrieval agent"
git push
```

## Hardware Choice

Start with CPU if you only need a simple demo. Use a GPU Space if:

- the first run takes too long,
- OpenCLIP image embedding is slow,
- you want to retrieve and rank more than roughly 100 candidates per prompt.

## Important Demo Notes

- The first run downloads the OpenCLIP model, so it will be slower.
- DuckDuckGo image search can occasionally rate-limit or return weak candidates.
- If the search layer is flaky, the architecture still holds: replace `search_image_candidates` with SerpAPI, Bing Image Search, or Google Custom Search.

## Best Prompt Settings

For stable demos:

- Candidate count: `80-120`
- Moodboard size: `20-30`
- Text weight without seed images: `0.8`
- Text weight with seed images: `0.55-0.7`
- Dedupe threshold: `0.90-0.94`

