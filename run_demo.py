#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from moodboard_agent.pipeline import run_moodboard_pipeline
from moodboard_agent.storage import save_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the moodboard agent prototype.")
    parser.add_argument("--directive", required=True, help="Art-style directive to collect references for.")
    parser.add_argument("--example", action="append", default=[], help="Optional seed example URL or note.")
    parser.add_argument("--count", type=int, default=36, help="Number of references to select.")
    args = parser.parse_args()

    run = run_moodboard_pipeline(args.directive, examples=args.example, target_count=args.count)
    path = save_run(run)

    print(f"Run id: {run.id}")
    print(f"Saved: {path}")
    print(f"Selected images: {len(run.selected_images)}")
    print(f"Rejected images: {len(run.rejected_images)}")
    print("Top selections:")
    for image in run.selected_images[:5]:
        score = image.score.total if image.score else 0
        print(f"- {image.title} | {image.source} | score={score}")


if __name__ == "__main__":
    main()

