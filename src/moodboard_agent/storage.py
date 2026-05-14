from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import MoodboardRun


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "data" / "sample_runs"
LATEST_PATH = RUN_DIR / "latest.json"


def save_run(run: MoodboardRun, path: Path | None = None) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    output_path = path or LATEST_PATH
    output_path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    return output_path


def load_latest_run() -> dict[str, Any] | None:
    if not LATEST_PATH.exists():
        return None
    return json.loads(LATEST_PATH.read_text(encoding="utf-8"))

