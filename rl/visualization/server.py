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
from rl.visualization.workflow_engine import WorkflowEngine

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMP_DIR = Path(__file__).resolve().parent / "temp"


class VisualizationRequestHandler(BaseHTTPRequestHandler):
    """Handles REST API and static asset requests for the RL audio dashboard."""

    engine: WorkflowEngine = WorkflowEngine()

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress noisy request logs in production, logging to debug instead."""
        addr = getattr(self, "client_address", ("127.0.0.1", 0))[0]
        logger.debug("%s - - [%s] %s", addr, self.log_date_time_string(), format % args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        """Helper to send JSON response."""
        encoded = json.dumps(data, indent=2).encode("utf-8")
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
            self._send_json(self.engine.get_summary())
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
            self.engine.reset_session()
            self._send_json(self.engine.get_summary())
        elif path == "/api/upload":
            self._handle_upload(payload)
        else:
            self._send_error(f"Unknown POST endpoint: {path}", status=404)

    # -----------------------------------------------------------------
    # Handler Implementations
    # -----------------------------------------------------------------
    def _handle_status(self) -> None:
        """Return server status and environment diagnostic details."""
        status_data = {
            "status": "online",
            "mode": self.engine.mode,
            "has_real_agent": self.engine.has_real_agent,
            "is_initialized": self.engine.is_initialized,
            "is_terminated": self.engine.is_terminated,
            "current_step": self.engine.current_step,
            "device": str(self.engine.device),
            "predictor_ckpt_exists": bool(self.engine.predictor_ckpt and self.engine.predictor_ckpt.exists()),
            "grrl_ckpt_exists": bool(self.engine.grrl_ckpt and self.engine.grrl_ckpt.exists()),
            "presets": DEFAULT_PRESETS,
        }
        self._send_json(status_data)

    def _handle_initialize(self, payload: dict[str, Any]) -> None:
        """Initialize session with specified audio files and execution mode."""
        dry_path = payload.get("dry_path")
        wet_path = payload.get("wet_path")
        mode = payload.get("mode", "real")
        use_warm_start = bool(payload.get("use_warm_start", False))
        max_steps = int(payload.get("max_steps", 8))

        # Default to first preset if not provided
        if not dry_path or not wet_path:
            dry_path = DEFAULT_PRESETS[0]["dry_path"]
            wet_path = DEFAULT_PRESETS[0]["wet_path"]

        try:
            summary = self.engine.initialize_session(
                dry_path=dry_path,
                wet_path=wet_path,
                mode=mode,
                use_warm_start=use_warm_start,
                max_steps=max_steps,
            )
            self._send_json(summary)
        except Exception as e:
            logger.exception("Initialization failed")
            self._send_error(f"Initialization failed: {e}", status=500)

    def _handle_step(self) -> None:
        """Advance one single action step."""
        try:
            if not self.engine.is_initialized:
                # Auto-initialize with default preset
                self.engine.initialize_session(
                    dry_path=DEFAULT_PRESETS[0]["dry_path"],
                    wet_path=DEFAULT_PRESETS[0]["wet_path"],
                    mode="real",
                )
            summary = self.engine.step_forward()
            self._send_json(summary)
        except Exception as e:
            logger.exception("Step execution failed")
            self._send_error(f"Step execution failed: {e}", status=500)

    def _handle_run_all(self) -> None:
        """Run workflow to completion."""
        try:
            if not self.engine.is_initialized:
                self.engine.initialize_session(
                    dry_path=DEFAULT_PRESETS[0]["dry_path"],
                    wet_path=DEFAULT_PRESETS[0]["wet_path"],
                    mode="real",
                )
            summary = self.engine.run_all()
            self._send_json(summary)
        except Exception as e:
            logger.exception("Run all execution failed")
            self._send_error(f"Run all failed: {e}", status=500)

    def _handle_upload(self, payload: dict[str, Any]) -> None:
        """Handle custom dry and wet audio base64 uploads."""
        dry_b64 = payload.get("dry_base64", "")
        wet_b64 = payload.get("wet_base64", "")
        mode = payload.get("mode", "real")

        if not dry_b64 or not wet_b64:
            self._send_error("Both dry_base64 and wet_base64 are required.")
            return

        try:
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            dry_bytes = base64.b64decode(dry_b64.split(",")[-1])
            wet_bytes = base64.b64decode(wet_b64.split(",")[-1])

            dry_file = TEMP_DIR / "uploaded_dry.wav"
            wet_file = TEMP_DIR / "uploaded_wet.wav"

            dry_file.write_bytes(dry_bytes)
            wet_file.write_bytes(wet_bytes)

            summary = self.engine.initialize_session(
                dry_path=dry_file,
                wet_path=wet_file,
                mode=mode,
            )
            self._send_json(summary)
        except Exception as e:
            logger.exception("Failed to process uploaded audio")
            self._send_error(f"Upload processing failed: {e}", status=500)

    def _handle_audio(self, query: dict[str, list[str]]) -> None:
        """Stream WAV audio for native browser playback."""
        audio_type = query.get("type", ["output"])[0]

        tensor = None
        if audio_type == "dry":
            tensor = self.engine.dry_tensor
        elif audio_type == "wet":
            tensor = self.engine.wet_tensor
        elif audio_type == "output":
            tensor = self.engine.final_audio_tensor or self.engine.current_audio_tensor

        if tensor is None:
            self._send_error(f"Audio type '{audio_type}' is not available yet.", status=404)
            return

        try:
            wav_bytes = tensor_to_wav_bytes(tensor, self.engine.sample_rate)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav_bytes)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(wav_bytes)
        except Exception as e:
            self._send_error(f"Failed to encode audio stream: {e}", status=500)

    def _handle_download(self, query: dict[str, list[str]]) -> None:
        """Download synthesized audio file as attachment."""
        audio_type = query.get("type", ["output"])[0]
        tensor = self.engine.final_audio_tensor or self.engine.current_audio_tensor

        if tensor is None:
            self._send_error("Synthesized audio is not available to download.", status=404)
            return

        try:
            wav_bytes = tensor_to_wav_bytes(tensor, self.engine.sample_rate)
            filename = f"dapn_{self.engine.mode}_{audio_type}.wav"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(wav_bytes)))
            self.end_headers()
            self.wfile.write(wav_bytes)
        except Exception as e:
            self._send_error(f"Download failed: {e}", status=500)

    def _handle_export_json(self) -> None:
        """Export session diagnostics and final FX chain as JSON file."""
        summary = self.engine.get_summary()
        encoded = json.dumps(summary, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Disposition", 'attachment; filename="rl_audio_workflow_report.json"')
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


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
    actual_port = find_available_port(host, port)
    server_address = (host, actual_port)
    httpd = ThreadingHTTPServer(server_address, VisualizationRequestHandler)

    url = f"http://{host}:{actual_port}"
    print("\n" + "=" * 70)
    print(" 🎛️  DAPN RL AUDIO SYNTHESIS VISUALIZATION STUDIO")
    print("=" * 70)
    print(f" Web Dashboard URL: {url}")
    print(f" Real Backend Available: {VisualizationRequestHandler.engine.has_real_agent}")
    print(f" Default Mode: {'REAL RL (GRRL Policy)' if VisualizationRequestHandler.engine.has_real_agent else 'DEMO (Simulated)'}")
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
