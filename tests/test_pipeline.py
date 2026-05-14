from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodboard_agent.pipeline import run_moodboard_pipeline
from moodboard_agent.schemas import MoodboardRequest
from moodboard_agent.style_analysis import analyze_style


class StyleAnalysisTests(unittest.TestCase):
    def test_known_style_extracts_specific_traits(self) -> None:
        brief = analyze_style(MoodboardRequest("Pixar style family adventure"))

        self.assertIn("expressive character design", brief.keywords)
        self.assertIn("warm saturated primaries", brief.palette)
        self.assertGreaterEqual(len(brief.search_queries), 4)


class PipelineTests(unittest.TestCase):
    def test_pipeline_selects_requested_count(self) -> None:
        run = run_moodboard_pipeline("Pixar style animated feature", target_count=12)

        self.assertEqual(len(run.selected_images), 12)
        self.assertGreater(len(run.rejected_images), 0)
        self.assertTrue(all(image.score for image in run.selected_images))

    def test_pipeline_includes_manual_example_source(self) -> None:
        run = run_moodboard_pipeline(
            "storybook miniature set design",
            examples=["https://example.com/manual-seed.jpg"],
            target_count=8,
        )

        sources = {plan.source for plan in run.source_plan}
        self.assertIn("Manual Example Expansion", sources)


if __name__ == "__main__":
    unittest.main()

