from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from .pipeline import run_moodboard_pipeline
from .storage import load_latest_run, save_run


ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = ROOT / "dashboard"


class MoodboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/dashboard":
            self._serve_file(DASHBOARD_DIR / "index.html")
            return
        if self.path == "/api/runs/latest":
            payload = load_latest_run()
            if payload is None:
                run = run_moodboard_pipeline("Pixar style animated feature, warm family adventure", target_count=36)
                save_run(run)
                payload = run.to_dict()
            self._send_json(payload)
            return
        if self.path.startswith("/dashboard/"):
            relative = unquote(self.path.removeprefix("/dashboard/"))
            self._serve_file(DASHBOARD_DIR / relative)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        directive = str(body.get("directive", "")).strip()
        if not directive:
            self.send_error(400, "Missing directive")
            return

        examples = body.get("examples", [])
        if isinstance(examples, str):
            examples = [line.strip() for line in examples.splitlines() if line.strip()]
        target_count = int(body.get("target_count", 36))

        run = run_moodboard_pipeline(directive, examples=examples, target_count=target_count)
        save_run(run)
        self._send_json(run.to_dict())

    def log_message(self, format: str, *args: object) -> None:
        return

    def _serve_file(self, path: Path) -> None:
        resolved = path.resolve()
        if not str(resolved).startswith(str(DASHBOARD_DIR.resolve())) or not resolved.exists() or resolved.is_dir():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        body = resolved.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    server = ThreadingHTTPServer((host, port), MoodboardHandler)
    print(f"Moodboard dashboard running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve()

