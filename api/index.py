"""INFRADIAN inference API as a Vercel Python serverless function.

Runs same-origin with the static frontend, so there is no CORS hop and no second host to keep
alive. vercel.json rewrites every /api/* path here.

Deliberately stdlib-only apart from numpy. Two reasons:
  - the training stack (pandas, scikit-learn, and scipy via lightgbm) installs to ~660MB, well
    over Vercel's 500MB function limit;
  - a web framework is one more dependency that can fail to install in the build image, which it
    did. `BaseHTTPRequestHandler` is guaranteed present.

Inference therefore goes through `_inference.py` (numpy reimplementation of the training feature
transform) and `_booster.py` (numpy reader for LightGBM's model text format). Both are proven
numerically identical to the training path in tests/test_serve_parity.py, so the deployed model
cannot silently differ from the evaluated one.

The model served is the synthetic-trained checkpoint, the only one we may legally distribute.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MODEL_DIR = HERE.parent / "results" / "models" / "infradian-ref-s"


def _model_available() -> bool:
    return (MODEL_DIR / "phase_clf.txt").exists()


def _healthz() -> tuple[int, dict]:
    return 200, {
        "status": "ok",
        "model_loaded": _model_available(),
        "runtime": "vercel-python",
    }


def _model_meta() -> tuple[int, dict]:
    meta = {}
    try:
        import _inference

        meta = _inference.load_models()["meta"]
    except Exception:  # noqa: BLE001 - metadata must never 500
        pass
    return 200, {
        "name": "infradian-ref-s",
        "trained_on": "synthetic (Tier C) only, the only legally distributable checkpoint",
        "license": "Apache-2.0",
        "arm": meta.get("arm"),
        "n_features": len(meta.get("feature_columns", [])),
        "serving_path": "numpy feature transform + numpy LightGBM reader, parity-tested against training",
    }


def _predict(body: dict) -> tuple[int, dict]:
    days = body.get("days") or []
    if not days:
        return 400, {"error": "no days provided"}
    if not _model_available():
        return 503, {"error": "reference model not available"}

    import _inference

    try:
        out = _inference.predict(days)
    except Exception as e:  # noqa: BLE001
        return 500, {"error": f"inference failed: {type(e).__name__}: {e}"}

    return 200, {
        "participant_id": body.get("participant_id", "user"),
        "model": "infradian-ref-s",
        **out,
    }


def _route(method: str, path: str, body: dict) -> tuple[int, dict]:
    path = path.rstrip("/") or "/"

    if path.endswith("/healthz"):
        return _healthz()
    if path.endswith("/model"):
        return _model_meta()
    if path.endswith("/predict/trajectory"):
        if method != "POST":
            return 405, {"error": "use POST"}
        return _predict(body)

    # --- multimodal AI surfaces. Imported lazily so a failure here can never stop the core
    # prediction API from serving. ---
    try:
        import _ai_routes as ai
    except Exception as e:  # noqa: BLE001
        return 503, {"error": f"AI routes unavailable: {type(e).__name__}: {e}"}

    if path.endswith("/ai/status"):
        return ai.ai_status()
    if path.endswith("/ai/vocabulary"):
        return ai.symptom_vocabulary()
    if path.endswith("/llm/journal"):
        if method != "POST":
            return 405, {"error": "use POST"}
        return ai.journal_extract(body)
    if path.endswith("/llm/transcribe"):
        if method != "POST":
            return 405, {"error": "use POST"}
        return ai.transcribe_audio(body)
    if path.endswith("/llm/read-strip"):
        if method != "POST":
            return 405, {"error": "use POST"}
        return ai.read_test_strip(body)
    if path.endswith("/llm/explain"):
        if method != "POST":
            return 405, {"error": "use POST"}
        return ai.explain(body)

    return 404, {"error": f"no route for {path}"}


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel requires this exact name
    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, {})

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        status, payload = _route("GET", path, {})
        self._send(status, payload)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON body"})
            return
        # `"hello"` and `[1,2]` are valid JSON but not objects, and every handler calls body.get(),
        # so they raised AttributeError and surfaced as a 500 on all four POST routes.
        if not isinstance(body, dict):
            self._send(400, {"error": "body must be a JSON object"})
            return
        status, payload = _route("POST", path, body)
        self._send(status, payload)

    def log_message(self, *args) -> None:  # keep the function logs quiet
        return
