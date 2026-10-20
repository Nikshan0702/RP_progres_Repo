"""
rl/visualization/server.py
==========================
Lightweight HTTP REST & Static File Server for RL Audio Processing Visualization.
Uses pure Python standard library http.server for zero external dependencies.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import socket
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from rl.visualization.audio_utils import tensor_to_wav_bytes
from rl.visualization.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_PRESETS,
    PROJECT_ROOT,
)
import threading

from rl.visualization.chart_exporter import export_svg_charts
from rl.visualization.workflow_engine import RealBackendUnavailable, WorkflowEngine, WorkflowStepError, json_safe

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMP_DIR = Path(__file__).resolve().parent / "temp"


class VisualizationRequestHandler(BaseHTTPRequestHandler):
    """Handles REST API and static asset requests for the RL audio dashboard."""

    # Created lazily (loading checkpoints at import time made the module untestable). Tests/CLI may assign it.
    engine: WorkflowEngine | None = None
    _lock = threading.RLock()

    @classmethod
    def get_engine(cls) -> WorkflowEngine:
        with cls._lock:
            if cls.engine is None:
                cls.engine = WorkflowEngine()
            return cls.engine

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress noisy request logs in production, logging to debug instead."""
        addr = getattr(self, "client_address", ("127.0.0.1", 0))[0]
        logger.debug("%s - - [%s] %s", addr, self.log_date_time_string(), format % args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        """Helper to send JSON response."""
        encoded = json.dumps(json_safe(data), indent=2, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error(self, message: str, status: int = 400) -> None:
        """Helper to send JSON error response."""
        self._send_json({"error": message, "status": status}, status=status)

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Route GET requests to API handlers or static files."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # -------------------------------------------------------------
        # REST API Routes
        # -------------------------------------------------------------
        if path == "/api/status":
            self._handle_status()
            return
        elif path == "/api/presets":
            self._send_json({"presets": DEFAULT_PRESETS})
            return
        elif path == "/api/summary":
            self._send_json(self.get_engine().get_summary())
            return
        elif path == "/api/audio":
            self._handle_audio(query)
            return
        elif path == "/api/download":
            self._handle_download(query)
            return
        elif path == "/api/export_json":
            self._handle_export_json()
            return
        elif path == "/api/export_svg":
            self._handle_export_svg()
            return

        # -------------------------------------------------------------
        # Static File Routes
        # -------------------------------------------------------------
        if path == "/" or path == "/index.html":
            file_path = STATIC_DIR / "index.html"
        elif path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            file_path = STATIC_DIR / rel_path
        else:
            file_path = STATIC_DIR / path.lstrip("/")

        if file_path.is_file() and file_path.exists():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                mime_type = "application/octet-stream"

            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self._send_error(f"Failed to read file: {e}", status=500)
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def do_POST(self) -> None:
        """Route POST requests for session initialization, stepping, and uploading."""
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            payload = json.loads(body) if body else {}
        except Exception as e:
            self._send_error(f"Invalid JSON request body: {e}")
            return

        if path == "/api/initialize":
            self._handle_initialize(payload)
        elif path == "/api/step":
            self._handle_step()
        elif path == "/api/run_all":
            self._handle_run_all()
        elif path == "/api/reset":
            self.get_engine().reset_session()
            self._send_json(self.get_engine().get_summary())
        elif path == "/api/upload":
            self._handle_upload(payload)
        else:
            self._send_error(f"Unknown POST endpoint: {path}", status=404)

    # -----------------------------------------------------------------
    # Handler Implementations
    # -----------------------------------------------------------------
    def _fail(self, exc: Exception, what: str) -> None:
        """Map failures to honest HTTP statuses. Real-mode unavailability is a 503 - never a demo fallback."""
        if isinstance(exc, RealBackendUnavailable):
            self._send_json({"error": str(exc), "code": "real_backend_unavailable", "status": 503,
                             "checkpoints": self.get_engine().checkpoint_status()}, status=503)
        elif isinstance(exc, (ValueError, FileNotFoundError)):
            self._send_json({"error": f"{what}: {exc}", "code": "bad_request", "status": 400}, status=400)
        elif isinstance(exc, WorkflowStepError):
            self._send_json({"error": f"{what}: {exc}", "code": "step_failed", "status": 500}, status=500)
        else:
            logger.exception(what)
            self._send_json({"error": f"{what}: {exc}", "code": "internal", "status": 500}, status=500)

    def _handle_status(self) -> None:
        eng = self.get_engine()
        self._send_json({
            "status": "online", "mode": eng.mode, "requested_mode": eng.requested_mode, "simulated": eng.mode == "demo",
            "has_real_agent": eng.has_real_agent, "backend_error": eng.backend_error,
            "predictor_available": eng.predictor_available, "predictor_error": eng.predictor_error,
            "is_initialized": eng.is_initialized, "is_terminated": eng.is_terminated, "current_step": eng.current_step,
            "device": str(eng.device), "checkpoints": eng.checkpoint_status(),
            "predictor_ckpt_exists": bool(eng.predictor_ckpt and eng.predictor_ckpt.exists()),
            "grrl_ckpt_exists": bool(eng.grrl_ckpt and eng.grrl_ckpt.exists()), "presets": DEFAULT_PRESETS})

    def _handle_initialize(self, payload: dict[str, Any]) -> None:
        dry_path, wet_path = payload.get("dry_path"), payload.get("wet_path")
        if not dry_path or not wet_path:
            dry_path, wet_path = DEFAULT_PRESETS[0]["dry_path"], DEFAULT_PRESETS[0]["wet_path"]
        try:
            with self._lock:
                self._send_json(self.get_engine().initialize_session(
                    dry_path=dry_path, wet_path=wet_path, mode=payload.get("mode", "real"),
                    use_warm_start=bool(payload.get("use_warm_start", False)), max_steps=int(payload.get("max_steps", 8)),
                    deterministic=bool(payload.get("deterministic", True))))
        except Exception as exc:
            self._fail(exc, "Initialization failed")

    def _ensure_session(self) -> None:
        if not self.get_engine().is_initialized:
            raise ValueError("No active session: POST /api/initialize first (no implicit default session is created)")

    def _handle_step(self) -> None:
        try:
            with self._lock:
                self._ensure_session()
                self._send_json(self.get_engine().step_forward())
        except Exception as exc:
            self._fail(exc, "Step execution failed")

    def _handle_run_all(self) -> None:
        try:
            with self._lock:
                self._ensure_session()
                self._send_json(self.get_engine().run_all())
        except Exception as exc:
            self._fail(exc, "Run all failed")

    def _handle_upload(self, payload: dict[str, Any]) -> None:
        dry_b64, wet_b64 = payload.get("dry_base64", ""), payload.get("wet_base64", "")
        if not dry_b64 or not wet_b64:
            self._send_error("Both dry_base64 and wet_base64 are required.")
            return
        try:
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            dry_file, wet_file = TEMP_DIR / "uploaded_dry.wav", TEMP_DIR / "uploaded_wet.wav"
            dry_file.write_bytes(base64.b64decode(dry_b64.split(",")[-1], validate=True))
            wet_file.write_bytes(base64.b64decode(wet_b64.split(",")[-1], validate=True))
            with self._lock:
                self._send_json(self.get_engine().initialize_session(
                    dry_path=dry_file, wet_path=wet_file, mode=payload.get("mode", "real"),
                    use_warm_start=bool(payload.get("use_warm_start", False)), max_steps=int(payload.get("max_steps", 8))))
        except Exception as exc:
            self._fail(exc, "Upload processing failed")

    def _pick_audio(self, audio_type: str):
        """Return (tensor | None, reason). Explicit None checks: tensors have no truth value."""
        eng = self.get_engine()
        if audio_type == "dry":
            return eng.dry_tensor, "dry audio not loaded"
        if audio_type == "wet":
            return eng.wet_tensor, "wet audio not loaded"
        if audio_type == "output":
            if eng.mode == "demo":
                return None, "demo mode is simulated: no audio was synthesized"
            if eng.final_audio_tensor is not None:
                return eng.final_audio_tensor, ""
            if eng.current_audio_tensor is not None:
                return eng.current_audio_tensor, ""   # in-progress render of the real chain so far
            return None, "no output audio yet"
        return None, f"unknown audio type '{audio_type}'"

    def _send_wav(self, tensor, filename: str | None, extra_headers: dict[str, str] | None = None) -> None:
        wav_bytes = tensor_to_wav_bytes(tensor, self.get_engine().sample_rate)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.send_header("Cache-Control", "no-cache")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        else:
            self.send_header("Accept-Ranges", "bytes")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(wav_bytes)

    def _handle_audio(self, query: dict[str, list[str]]) -> None:
        audio_type = query.get("type", ["output"])[0]
        tensor, reason = self._pick_audio(audio_type)
        if tensor is None:
            self._send_error(f"Audio '{audio_type}' unavailable: {reason}", status=404)
            return
        try:
            self._send_wav(tensor, None)
        except Exception as exc:
            self._send_error(f"Failed to encode audio stream: {exc}", status=500)

    def _handle_download(self, query: dict[str, list[str]]) -> None:
        audio_type = query.get("type", ["output"])[0]
        tensor, reason = self._pick_audio(audio_type)
        if tensor is None:
            self._send_error(f"Nothing to download: {reason}", status=409 if self.get_engine().mode == "demo" else 404)
            return
        eng = self.get_engine()
        try:
            self._send_wav(tensor, f"dapn_{eng.mode}_{audio_type}.wav", {"X-Execution-Mode": eng.mode})
        except Exception as exc:
            self._send_error(f"Download failed: {exc}", status=500)

    def _export_summary(self) -> dict[str, Any]:
        summary = self.get_engine().get_summary()
        for st in summary["stages"]:
            st["data"] = {k: v for k, v in st["data"].items() if k != "base64_audio"}   # keep the report compact
        summary["exported_at_unix"] = round(__import__("time").time(), 3)
        return summary

    def _handle_export_json(self) -> None:
        encoded = json.dumps(json_safe(self._export_summary()), indent=2, allow_nan=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Disposition", 'attachment; filename="rl_audio_workflow_report.json"')
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_export_svg(self) -> None:
        import tempfile
        eng = self.get_engine()
        if not eng.is_initialized:
            self._send_error("No active session to export", status=409)
            return
        with tempfile.TemporaryDirectory() as td:
            path = export_svg_charts(json_safe(eng.get_summary()), Path(td) / "charts.svg")
            data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/svg+xml")
        self.send_header("Content-Disposition", 'attachment; filename="rl_audio_charts.svg"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def find_available_port(host: str, start_port: int, max_attempts: int = 20) -> int:
    """Find next free socket port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return start_port


def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = False,
) -> None:
    """Start visualization web server."""
    VisualizationRequestHandler.get_engine()
    actual_port = find_available_port(host, port)
    server_address = (host, actual_port)
    httpd = ThreadingHTTPServer(server_address, VisualizationRequestHandler)

    url = f"http://{host}:{actual_port}"
    print("\n" + "=" * 70)
    print(" 🎛️  DAPN RL AUDIO SYNTHESIS VISUALIZATION STUDIO")
    print("=" * 70)
    print(f" Web Dashboard URL: {url}")
    print(f" Real Backend Available: {VisualizationRequestHandler.get_engine().has_real_agent}")
    _e = VisualizationRequestHandler.get_engine()
    print(f" Real mode: {'AVAILABLE' if _e.has_real_agent else 'UNAVAILABLE - ' + str(_e.backend_error)}")
    print(" Demo mode (scripted, SIMULATED numbers) must be selected explicitly.")
    print(" Press Ctrl+C to stop the server.")
    print("=" * 70 + "\n")

    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down visualization server...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the RL Audio Processing Visualization Server")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host address to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)
