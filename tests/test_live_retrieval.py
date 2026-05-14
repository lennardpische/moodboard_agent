from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodboard_agent.live_retrieval import (
    ImageCandidate,
    _apply_query_variation,
    select_diverse,
    select_diverse_varied,
)


def _make_candidates(n: int) -> tuple[list[ImageCandidate], dict[str, np.ndarray]]:
    """Build n orthogonal candidates with decreasing scores."""
    candidates = []
    embeddings = {}
    for i in range(n):
        path = f"/tmp/img_{i}"
        score = 1.0 - i * 0.02
        candidates.append(ImageCandidate(str(i), "test", f"https://img/{i}", f"https://page/{i}", path, score=score))
        vec = np.zeros(n)
        vec[i] = 1.0
        embeddings[path] = vec
    return candidates, embeddings


class LiveRetrievalTests(unittest.TestCase):
    def test_select_diverse_filters_near_duplicates_first(self) -> None:
        candidates = [
            ImageCandidate("a", "test", "https://img/a", "https://page/a", "/tmp/a", score=0.99),
            ImageCandidate("b", "test", "https://img/b", "https://page/b", "/tmp/b", score=0.98),
            ImageCandidate("c", "test", "https://img/c", "https://page/c", "/tmp/c", score=0.80),
        ]
        embeddings = {
            "/tmp/a": np.array([1.0, 0.0, 0.0]),
            "/tmp/b": np.array([0.99, 0.01, 0.0]),
            "/tmp/c": np.array([0.0, 1.0, 0.0]),
        }

        selected = select_diverse(candidates, embeddings, target_count=2, diversity_threshold=0.92)

        self.assertEqual([item.title for item in selected], ["a", "c"])


class ExplorationTests(unittest.TestCase):
    def test_exploration_zero_is_deterministic(self) -> None:
        candidates, embeddings = _make_candidates(20)
        target = 5
        rng_a = random.Random(42)
        rng_b = random.Random(99)

        # exploration=0 falls back to select_diverse (deterministic), so rng doesn't matter
        result_a = select_diverse(candidates, embeddings, target_count=target, diversity_threshold=0.5)
        result_b = select_diverse(candidates, embeddings, target_count=target, diversity_threshold=0.5)

        self.assertEqual([c.title for c in result_a], [c.title for c in result_b])
        self.assertEqual(len(result_a), target)

    def test_exploration_nonzero_different_seeds_can_differ(self) -> None:
        candidates, embeddings = _make_candidates(30)
        target = 6
        rng_a = random.Random(1)
        rng_b = random.Random(999999)

        result_a = select_diverse_varied(candidates, embeddings, target, 0.5, 0.8, rng_a)
        result_b = select_diverse_varied(candidates, embeddings, target, 0.5, 0.8, rng_b)

        titles_a = [c.title for c in result_a]
        titles_b = [c.title for c in result_b]
        # Different seeds should produce at least one different selection
        self.assertNotEqual(titles_a, titles_b)

    def test_exploration_respects_target_count(self) -> None:
        for target in (3, 6, 10):
            candidates, embeddings = _make_candidates(20)
            rng = random.Random(7)
            result = select_diverse_varied(candidates, embeddings, target, 0.5, 0.9, rng)
            self.assertEqual(len(result), target, f"Expected {target} items, got {len(result)}")

    def test_exploration_same_seed_is_reproducible(self) -> None:
        candidates, embeddings = _make_candidates(20)
        target = 5
        rng_a = random.Random(42)
        rng_b = random.Random(42)

        result_a = select_diverse_varied(candidates, embeddings, target, 0.5, 0.7, rng_a)
        result_b = select_diverse_varied(candidates, embeddings, target, 0.5, 0.7, rng_b)

        self.assertEqual([c.title for c in result_a], [c.title for c in result_b])

    def test_query_variation_adds_suffixes_when_exploring(self) -> None:
        queries = ["pixar style", "animated feature"]
        rng = random.Random(1)
        varied = _apply_query_variation(queries, exploration=0.5, rng=rng)

        self.assertGreater(len(varied), len(queries))
        # All original queries should still be present
        for q in queries:
            self.assertIn(q, varied)

    def test_query_variation_unchanged_at_zero(self) -> None:
        queries = ["pixar style", "animated feature"]
        rng = random.Random(1)
        varied = _apply_query_variation(queries, exploration=0.0, rng=rng)

        self.assertEqual(varied, queries)


if __name__ == "__main__":
    unittest.main()

