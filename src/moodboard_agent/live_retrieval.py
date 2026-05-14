from __future__ import annotations

import hashlib
import math
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .schemas import MoodboardRequest, StyleBrief
from .style_analysis import analyze_style


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


@dataclass
class RetrievalConfig:
    target_count: int = 12
    candidate_count: int = 40
    text_weight: float = 0.7
    diversity_threshold: float = 0.92
    request_timeout_seconds: int = 12
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"


@dataclass
class ImageCandidate:
    title: str
    source: str
    image_url: str
    page_url: str
    local_path: str
    score: float = 0.0


@dataclass
class RetrievalResult:
    brief: StyleBrief
    selected: list[ImageCandidate]
    raw_candidate_count: int
    downloaded_count: int
    model_name: str


class RetrievalDependencyError(RuntimeError):
    pass


def run_live_retrieval(
    directive: str,
    seed_image_paths: list[str] | None = None,
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    active_config = config or RetrievalConfig()
    request = MoodboardRequest(directive=directive, examples=seed_image_paths or [], target_count=active_config.target_count)
    brief = analyze_style(request)

    raw_candidates = search_image_candidates(brief.search_queries, active_config.candidate_count)
    downloaded = download_candidates(raw_candidates, active_config)
    if not downloaded:
        raise RuntimeError("No usable images were downloaded. Try a more specific prompt or rerun the search.")

    embedder = get_embedder(active_config.model_name, active_config.pretrained)
    text_embedding = embedder.embed_text(_retrieval_prompt(brief))
    image_embeddings = embedder.embed_images([candidate.local_path for candidate in downloaded])

    seed_embedding = None
    if seed_image_paths:
        valid_seed_paths = [path for path in seed_image_paths if path and Path(path).exists()]
        if valid_seed_paths:
            seed_vectors = embedder.embed_images(valid_seed_paths)
            seed_embedding = _normalize(seed_vectors.mean(axis=0, keepdims=True))[0]

    embeddings_by_path = {
        candidate.local_path: embedding
        for candidate, embedding in zip(downloaded, image_embeddings, strict=True)
    }
    ranked = rank_candidates(downloaded, image_embeddings, text_embedding, seed_embedding, active_config)
    selected = select_diverse(ranked, embeddings_by_path, active_config.target_count, active_config.diversity_threshold)

    return RetrievalResult(
        brief=brief,
        selected=selected,
        raw_candidate_count=len(raw_candidates),
        downloaded_count=len(downloaded),
        model_name=f"OpenCLIP {active_config.model_name} / {active_config.pretrained}",
    )


def search_image_candidates(queries: list[str], candidate_count: int) -> list[dict[str, Any]]:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise RetrievalDependencyError("Install ddgs with `pip install -r requirements.txt`.") from exc

    per_query = max(8, math.ceil(candidate_count / max(1, len(queries))))
    seen: set[str] = set()
    results: list[dict[str, Any]] = []

    with DDGS() as ddgs:
        for query in queries:
            if len(results) >= candidate_count:
                break
            try:
                for item in ddgs.images(query, max_results=per_query, safesearch="moderate"):
                    image_url = item.get("image") or item.get("thumbnail")
                    if not image_url or image_url in seen:
                        continue
                    seen.add(image_url)
                    results.append(
                        {
                            "title": item.get("title") or query,
                            "image_url": image_url,
                            "page_url": item.get("url") or item.get("source") or image_url,
                            "source": _domain(item.get("url") or image_url),
                        }
                    )
                    if len(results) >= candidate_count:
                        break
            except Exception:
                continue
            time.sleep(0.25)

    return results


def download_candidates(raw_candidates: list[dict[str, Any]], config: RetrievalConfig) -> list[ImageCandidate]:
    try:
        import requests
        from PIL import Image
    except ImportError as exc:
        raise RetrievalDependencyError("Install pillow and requests with `pip install -r requirements.txt`.") from exc

    output_dir = Path(tempfile.mkdtemp(prefix="moodboard_images_"))
    downloaded: list[ImageCandidate] = []

    for raw in raw_candidates:
        image_url = raw["image_url"]
        try:
            response = requests.get(
                image_url,
                timeout=config.request_timeout_seconds,
                headers={"User-Agent": USER_AGENT},
                stream=True,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "image" not in content_type:
                continue

            suffix = _image_suffix(content_type, image_url)
            local_path = output_dir / f"{_url_hash(image_url)}{suffix}"
            local_path.write_bytes(response.content)

            with Image.open(local_path) as image:
                image.verify()
            with Image.open(local_path) as image:
                width, height = image.size
            if width < 160 or height < 160:
                continue

            downloaded.append(
                ImageCandidate(
                    title=raw["title"],
                    source=raw["source"],
                    image_url=image_url,
                    page_url=raw["page_url"],
                    local_path=str(local_path),
                )
            )
        except Exception:
            continue

    return downloaded


def rank_candidates(
    candidates: list[ImageCandidate],
    image_embeddings: np.ndarray,
    text_embedding: np.ndarray,
    seed_embedding: np.ndarray | None,
    config: RetrievalConfig,
) -> list[ImageCandidate]:
    text_scores = image_embeddings @ text_embedding
    if seed_embedding is not None:
        seed_scores = image_embeddings @ seed_embedding
        scores = config.text_weight * text_scores + (1.0 - config.text_weight) * seed_scores
    else:
        scores = text_scores

    for candidate, score in zip(candidates, scores, strict=True):
        candidate.score = float(score)

    return sorted(candidates, key=lambda item: item.score, reverse=True)


def select_diverse(
    ranked: list[ImageCandidate],
    embeddings_by_path: dict[str, np.ndarray],
    target_count: int,
    diversity_threshold: float,
) -> list[ImageCandidate]:
    selected: list[ImageCandidate] = []
    selected_vectors: list[np.ndarray] = []

    for candidate in ranked:
        vector = embeddings_by_path[candidate.local_path]
        if not selected_vectors:
            selected.append(candidate)
            selected_vectors.append(vector)
        else:
            similarities = [float(vector @ selected_vector) for selected_vector in selected_vectors]
            if max(similarities) < diversity_threshold:
                selected.append(candidate)
                selected_vectors.append(vector)

        if len(selected) >= target_count:
            return selected

    selected_paths = {candidate.local_path for candidate in selected}
    for candidate in ranked:
        if len(selected) >= target_count:
            break
        if candidate.local_path not in selected_paths:
            selected.append(candidate)
            selected_paths.add(candidate.local_path)

    return selected


class OpenClipEmbedder:
    def __init__(self, model_name: str, pretrained: str) -> None:
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RetrievalDependencyError("Install OpenCLIP dependencies with `pip install -r requirements.txt`.") from exc

        self.open_clip = open_clip
        self.torch = torch
        self.image_cls = Image
        self.device = _device(torch)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def embed_text(self, text: str) -> np.ndarray:
        with self.torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            embedding = self.model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.detach().cpu().numpy()[0]

    def embed_images(self, paths: list[str], batch_size: int = 16) -> np.ndarray:
        vectors = []
        with self.torch.no_grad():
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                images = []
                for path in batch_paths:
                    image = self.image_cls.open(path).convert("RGB")
                    images.append(self.preprocess(image))
                tensor = self.torch.stack(images).to(self.device)
                embedding = self.model.encode_image(tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                vectors.append(embedding.detach().cpu().numpy())
        return np.concatenate(vectors, axis=0)


@lru_cache(maxsize=2)
def get_embedder(model_name: str, pretrained: str) -> OpenClipEmbedder:
    return OpenClipEmbedder(model_name, pretrained)


def _retrieval_prompt(brief: StyleBrief) -> str:
    return " ".join(
        [
            brief.directive,
            "visual moodboard reference",
            "keywords:",
            ", ".join(brief.keywords),
            "palette:",
            ", ".join(brief.palette),
            "lighting:",
            ", ".join(brief.lighting),
            "composition:",
            ", ".join(brief.composition),
        ]
    )


def _device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)


def _url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "") or "image-search"


def _image_suffix(content_type: str, url: str) -> str:
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "gif" in content_type:
        return ".gif"
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return suffix
    return ".jpg"
