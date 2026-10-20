"""
rl/test_visualization.py
========================
Automated test suite for the RL Audio Processing Visualization components.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from dapn.dapn import DAPN
from rl.visualization.audio_utils import (
    compute_audio_metrics,
    compute_waveform_peaks,
    load_audio_from_path,
    tensor_to_base64_wav,
    tensor_to_wav_bytes,
)
from rl.visualization.config import (
    DEFAULT_PRESETS,
    WORKFLOW_STAGES,
    get_effects_spec_map,
)
from rl.visualization.server import VisualizationRequestHandler
from rl.visualization.state_extractor import (
    compute_reward_breakdown,
    extract_action_diagnostics,
    extract_observation_diagnostics,
    map_parameters_with_physical,
)
from rl.visualization.workflow_engine import WorkflowEngine


class TestAudioUtils:
    """Test suite for audio loading, metrics, downsampling, and base64 encoding."""

    def test_load_audio_formatting(self):
        dry_path = DEFAULT_PRESETS[0]["dry_path"]
        waveform, sr = load_audio_from_path(dry_path, target_sample_rate=44100)
        assert waveform.ndim == 3, f"Expected shape (1, 1, T), got {waveform.shape}"
        assert waveform.shape[0] == 1 and waveform.shape[1] == 1
        assert sr == 44100
        assert torch.isfinite(waveform).all()

    def test_compute_waveform_peaks(self):
        t = torch.sin(torch.linspace(0, 100, 4410)) * 0.8
        peaks = compute_waveform_peaks(t, num_points=50)

        assert "min_peaks" in peaks and "max_peaks" in peaks and "rms" in peaks
        assert len(peaks["min_peaks"]) == 50
        assert len(peaks["max_peaks"]) == 50
        assert len(peaks["rms"]) == 50

        # Check bounded values
        assert all(0.0 <= v <= 1.0 for v in peaks["max_peaks"])
        assert all(-1.0 <= v <= 0.0 for v in peaks["min_peaks"])

    def test_compute_audio_metrics(self):
        t = torch.randn(1, 1, 22050) * 0.5
        metrics = compute_audio_metrics(t, sample_rate=44100)

        assert metrics["sample_rate"] == 44100
        assert metrics["total_samples"] == 22050
        assert metrics["duration_sec"] == 0.5
        assert metrics["peak_amplitude"] > 0
        assert metrics["rms_amplitude"] > 0

    def test_tensor_to_wav_bytes_and_base64(self):
        t = torch.randn(1, 1, 1000) * 0.3
        wav_bytes = tensor_to_wav_bytes(t, sample_rate=44100)
        assert len(wav_bytes) > 44  # WAV header is 44 bytes
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"

        b64 = tensor_to_base64_wav(t, sample_rate=44100)
        assert b64.startswith("data:audio/wav;base64,")


class TestStateExtractor:
    """Test suite for state observation, parameter mapping, and action extraction."""

    def test_get_effects_spec_map(self):
        specs = get_effects_spec_map()
        for eff in ["EQ", "Compressor", "Reverb", "Distortion", "Delay", "Chorus", "Flanger"]:
            assert eff in specs
            assert "parameters" in specs[eff]

    def test_map_parameters_with_physical(self):
        # EQ has frequency [20, 20000] Hz, gain [-12, 12] dB, q [0.1, 10]
        params = {"frequency": 0.5, "gain": 0.25, "q": 0.8}
        prev_params = {"frequency": 0.2, "gain": 0.25, "q": 0.5}

        mapped = map_parameters_with_physical("EQ", params, prev_params)
        assert len(mapped) == 3

        freq_item = next(item for item in mapped if item["name"] == "frequency")
        assert freq_item["unit"] == "Hz"
        assert freq_item["normalized"] == 0.5
        # Midpoint of [20, 20000] is 20 + 0.5 * 19980 = 10010 Hz
        assert 10000 < freq_item["physical"] < 10020
        assert freq_item["delta_normalized"] == pytest.approx(0.3, abs=1e-4)

    def test_extract_observation_diagnostics(self):
        obs = torch.randn(188)
        diag = extract_observation_diagnostics(obs)

        assert diag["total_dim"] == 188
        assert len(diag["slot_parameters"]) == 8
        assert len(diag["acoustic_profile"]) == 64
        assert len(diag["chain_history_slots"]) == 8
        assert "step_progress_pct" in diag["metrics"]
        assert len(diag["taste_vector"]) == 16

    def test_extract_action_diagnostics(self):
        logits = torch.tensor([1.0, 2.0, 0.5, 3.5, 0.2, 0.1, 0.1, -1.0, 0.0])
        mask = torch.tensor([True, True, True, True, True, True, True, False, True])
        diag = extract_action_diagnostics(logits, mask, chosen_action_idx=3, value_estimate=1.25)

        assert diag["chosen_action_idx"] == 3
        assert diag["chosen_action"] == "Reverb"
        assert diag["value_estimate"] == 1.25
        assert len(diag["distribution"]) == 9

        stop_action = diag["distribution"][7]
        assert stop_action["is_valid"] is False
        assert stop_action["mask_reason"] != ""

    def test_compute_reward_breakdown(self):
        rb = compute_reward_breakdown(
            prev_loss=5.0,
            current_loss=3.5,
            initial_loss=7.0,
            is_terminal=False,
            target_achieved=False,
            step_penalty=0.05,
        )
        assert rb["delta_loss"] == pytest.approx(1.5, abs=1e-4)
        assert rb["loss_improvement_reward"] == pytest.approx(15.0, abs=1e-4)
        assert rb["step_penalty"] == -0.05
        assert rb["total_step_reward"] == pytest.approx(14.95, abs=1e-4)


class TestWorkflowEngine:
    """Test suite for 11-stage workflow progression in both Real and Demo modes."""

    def test_workflow_engine_demo_progression(self):
        engine = WorkflowEngine()
        dry_path = DEFAULT_PRESETS[0]["dry_path"]
        wet_path = DEFAULT_PRESETS[0]["wet_path"]

        summary = engine.initialize_session(
            dry_path=dry_path,
            wet_path=wet_path,
            mode="demo",
        )
        assert summary["mode"] == "demo"
        assert summary["is_initialized"] is True
        assert summary["stages"][0]["status"] == "completed"  # upload
        assert summary["stages"][1]["status"] == "completed"  # preprocess

        # Step forward
        step1 = engine.step_forward()
        assert step1["current_step"] == 1
        assert len(step1["steps_history"]) == 1
        assert step1["steps_history"][0]["action"] == "Distortion"

        # Complete run
        final_summary = engine.run_all()
        assert final_summary["is_terminated"] is True
        assert final_summary["current_loss"] < final_summary["initial_loss"]
        assert len(final_summary["current_chain"]) > 0

        # Check all 11 stages completed
        completed_stage_ids = {s["id"] for s in final_summary["stages"] if s["status"] == "completed"}
        for stage in WORKFLOW_STAGES:
            assert stage["id"] in completed_stage_ids

    def test_workflow_engine_real_backend(self):
        engine = WorkflowEngine()
        if not engine.has_real_agent:
            pytest.skip("Trained schema-v2 GRRL checkpoint not present on disk.")
        summary = engine.initialize_session(DEFAULT_PRESETS[0]["dry_path"], DEFAULT_PRESETS[0]["wet_path"], mode="real", max_steps=5)
        assert summary["mode"] == "real" and summary["is_initialized"] is True
        final = engine.run_all()
        # A trained checkpoint may or may not improve on a given pair; the dashboard must report the truth.
        assert final["is_terminated"] is True and final["current_step"] >= 1
        assert engine.final_audio_tensor is not None and torch.isfinite(engine.final_audio_tensor).all()
        assert final["verification"]["chain_matches_audio"] is True
        expected = 100 * (final["start_loss"] - final["current_loss"]) / max(abs(final["start_loss"]), 1e-8)
        assert final["improvement_pct"] == pytest.approx(expected, abs=1e-3)


class TestServerRequestHandler:
    """Test suite for server request routing and API responses."""

    def test_direct_handler_routing(self):
        class DirectHandler(VisualizationRequestHandler):
            def __init__(self, method, path, body=b""):
                self.rfile = io.BytesIO(body)
                self.wfile = io.BytesIO()
                self.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
                self.command = method
                self.path = path
                self.requestline = f"{method} {path} HTTP/1.1"
                self.request_version = "HTTP/1.1"
                self.close_connection = False

        # Test GET /api/status
        h_status = DirectHandler("GET", "/api/status")
        h_status.do_GET()
        resp_status = h_status.wfile.getvalue().decode("utf-8")
        assert "200 OK" in resp_status
        assert "online" in resp_status

        # Test GET / (HTML index)
        h_index = DirectHandler("GET", "/")
        h_index.do_GET()
        resp_index = h_index.wfile.getvalue().decode("utf-8")
        assert "200 OK" in resp_index
        assert "DAPN RL Audio Synthesis Studio" in resp_index

        # Test POST /api/initialize
        VisualizationRequestHandler.engine = WorkflowEngine()
        h_init = DirectHandler("POST", "/api/initialize", json.dumps({"mode": "demo"}).encode("utf-8"))
        h_init.do_POST()
        resp_init = h_init.wfile.getvalue().decode("utf-8")
        assert "200 OK" in resp_init
        assert "is_initialized" in resp_init

        # Test POST /api/step
        h_step = DirectHandler("POST", "/api/step", b"{}")
        h_step.do_POST()
        resp_step = h_step.wfile.getvalue().decode("utf-8")
        assert "200 OK" in resp_step
        assert "current_step" in resp_step

        # Test GET /api/audio?type=dry
        h_audio = DirectHandler("GET", "/api/audio?type=dry")
        h_audio.do_GET()
        resp_audio = h_audio.wfile.getvalue()
        assert b"200 OK" in resp_audio
        assert b"audio/wav" in resp_audio

    def test_dapn_dsp_pipeline_integrity(self):
        """Ensure that existing DAPN DSP functionality remains intact."""
        dapn = DAPN()
        dry = torch.randn(1, 1, 22050)
        chain = {
            "chain": [
                {"effect": "EQ", "parameters": {"frequency": 0.5, "gain": 0.6, "q": 0.5}},
                {"effect": "Distortion", "parameters": {"drive": 0.3, "tone": 0.5, "mix": 0.5, "output_gain": 0.5}},
            ]
        }
        out = dapn.process(dry, chain)
        assert out.shape == dry.shape
        assert torch.isfinite(out).all()
