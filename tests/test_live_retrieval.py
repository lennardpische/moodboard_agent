from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodboard_agent.live_retrieval import ImageCandidate, select_diverse


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


if __name__ == "__main__":
    unittest.main()

