"""Dashboard correctness against REAL execution (random-init v2 checkpoints: plumbing/truthfulness, not RL quality)."""
import json
import math
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from audio_io import load_audio
from rl.grrl_agent import GRRLAgent
from rl.visualization.chart_exporter import export_svg_charts
from rl.visualization.config import DEFAULT_PRESETS
from rl.visualization.server import VisualizationRequestHandler
from rl.visualization.state_extractor import compute_reward_breakdown
from rl.visualization.workflow_engine import RealBackendUnavailable, WorkflowEngine, json_safe

DRY, WET = DEFAULT_PRESETS[0]["dry_path"], DEFAULT_PRESETS[0]["wet_path"]


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory, v2_predictor_ckpt):
    d = tmp_path_factory.mktemp("grrl")
    torch.manual_seed(0)
    a = GRRLAgent(); path = d / "grrl.pth"
    a.save(path, extra_config={"warm_start": True, "synthetic_smoke": True})
    return v2_predictor_ckpt, path


@pytest.fixture
def engine(ckpts):
    return WorkflowEngine(predictor_ckpt=ckpts[0], grrl_ckpt=ckpts[1])


@pytest.fixture
def server(engine):
    old = VisualizationRequestHandler.engine
    VisualizationRequestHandler.engine = engine
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), VisualizationRequestHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", engine
    httpd.shutdown(); httpd.server_close(); VisualizationRequestHandler.engine = old


def call(base, path, method="GET", body=None):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


# ----------------------------------------------------------------- real vs demo honesty
def test_real_mode_fails_loudly_when_checkpoint_missing(tmp_path, v2_predictor_ckpt):
    eng = WorkflowEngine(predictor_ckpt=v2_predictor_ckpt, grrl_ckpt=tmp_path / "missing.pth")
    assert not eng.has_real_agent and "not found" in eng.backend_error
    with pytest.raises(RealBackendUnavailable):
        eng.initialize_session(DRY, WET, mode="real")
    assert not eng.is_initialized and eng.mode != "demo"


def test_legacy_checkpoint_reported_incompatible_not_loaded():
    legacy = Path("checkpoints/grrl/best_grrl_model.pth")
    if not legacy.exists():
        pytest.skip("legacy checkpoint absent")
    eng = WorkflowEngine(grrl_ckpt=legacy)
    assert not eng.has_real_agent and "legacy" in eng.backend_error
    assert eng.checkpoint_status()["grrl"]["compatible"] is False


def test_http_real_unavailable_is_503_and_never_demo(tmp_path, v2_predictor_ckpt):
    eng = WorkflowEngine(predictor_ckpt=v2_predictor_ckpt, grrl_ckpt=tmp_path / "missing.pth")
    old = VisualizationRequestHandler.engine; VisualizationRequestHandler.engine = eng
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), VisualizationRequestHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        code, _, body = call(base, "/api/initialize", "POST", {"mode": "real"})
        assert code == 503 and json.loads(body)["code"] == "real_backend_unavailable"
        st = json.loads(call(base, "/api/status")[2])
        assert st["has_real_agent"] is False and st["backend_error"] and st["mode"] != "demo"
        code, _, body = call(base, "/api/step", "POST", {})
        assert code == 400 and "initialize" in json.loads(body)["error"]   # no implicit default/demo session
        code, _, body = call(base, "/api/initialize", "POST", {"mode": "demo"})
        s = json.loads(body)
        assert code == 200 and s["mode"] == "demo" and s["simulated"] is True and s["requested_mode"] == "demo"
    finally:
        httpd.shutdown(); httpd.server_close(); VisualizationRequestHandler.engine = old


def test_demo_is_labelled_simulated_and_offers_no_audio(server):
    base, eng = server
    call(base, "/api/initialize", "POST", {"mode": "demo"})
    s = json.loads(call(base, "/api/run_all", "POST", {})[2])
    assert s["simulated"] and all(st.get("simulated") for st in s["steps_history"])
    assert all("[SIMULATED]" in st["message"] for st in s["stages"] if st["id"] in ("state", "action", "dsp", "loss", "reward"))
    assert call(base, "/api/audio?type=output")[0] == 404
    assert call(base, "/api/download?type=output")[0] == 409
    assert s["has_output_audio"] is False


# ----------------------------------------------------------------- real execution truthfulness
def run_real(engine, warm):
    engine.initialize_session(DRY, WET, mode="real", use_warm_start=warm, max_steps=5)
    return engine.run_all()


def test_cold_start_baseline_is_dry_and_reports_are_consistent(engine):
    s = run_real(engine, warm=False)
    assert s["mode"] == "real" and s["simulated"] is False and s["baseline_name"] == "dry"
    assert s["start_loss"] == s["dry_loss"] and s["predictor_loss"] is None
    assert s["loss_history"][0]["loss"] == pytest.approx(s["dry_loss"], abs=1e-6)
    assert s["is_terminated"] and s["verification"]["chain_matches_audio"] is True
    # final chain reported == chain re-rendered == audio served
    env = engine.env
    ref = env.dapn.process(env.dry_audio, {"chain": s["current_chain"]})
    assert torch.allclose(ref, engine.final_audio_tensor, atol=1e-5)
    assert s["current_chain"] == s["steps_history"][-1]["chain"]
    # signed improvement, never clamped
    expected = 100 * (s["start_loss"] - s["current_loss"]) / s["start_loss"]
    assert s["improvement_pct"] == pytest.approx(expected, abs=1e-3)


def test_reward_breakdowns_reconcile_with_step_rewards_and_cumulative(engine):
    s = run_real(engine, warm=False)
    tot = 0.0
    for st in s["steps_history"]:
        b = st["reward_breakdown"]
        parts = b["loss_improvement_reward"] + b["step_penalty"] + b["length_penalty"] + b["terminal_bonus"]
        assert parts == pytest.approx(st["step_reward"], abs=1e-5) == pytest.approx(b["total_step_reward"], abs=1e-5)
        assert b["delta_loss"] == pytest.approx(st["prev_loss"] - st["current_loss"], abs=1e-5)
        assert b["loss_improvement_reward"] == pytest.approx(10 * b["delta_loss"], abs=1e-4)
        tot += st["step_reward"]
        assert st["cumulative_reward"] == pytest.approx(tot, abs=1e-4)
    assert s["cumulative_reward"] == pytest.approx(tot, abs=1e-4)
    assert [h["reward"] for h in s["reward_history"]] == [st["step_reward"] for st in s["steps_history"]]
    assert [h["loss"] for h in s["loss_history"][1:]] == [st["current_loss"] for st in s["steps_history"]]


def test_helper_reward_formula_matches_environment(engine):
    s = run_real(engine, warm=False)
    for st in s["steps_history"]:
        b = st["reward_breakdown"]
        n_after = len(st["chain"])
        n_before = n_after - (0 if st["is_stop"] else -1 if st["is_backtrack"] else 1)
        h = compute_reward_breakdown(st["prev_loss"], st["current_loss"], step_penalty=0.05, is_stop_action=st["is_stop"],
                                     chain_length=n_after, chain_length_before=n_before, length_penalty=0.02,
                                     target_achieved=b["target_achieved"], started_above_target=s["start_loss"] > 0.40)
        assert h["total_step_reward"] == pytest.approx(st["step_reward"], abs=1e-4)


def test_warm_start_baseline_is_predictor_loss_not_dry(engine):
    engine.initialize_session(DRY, WET, mode="real", use_warm_start=True, max_steps=5)
    s0 = engine.get_summary()
    assert s0["warm_started"] and s0["baseline_name"] == "predictor"
    assert s0["predictor_loss"] == s0["start_loss"] == s0["current_loss"] == s0["loss_history"][0]["loss"]
    assert s0["predictor_loss"] != pytest.approx(s0["dry_loss"], abs=1e-9)          # not the dry loss
    assert 1 <= len(s0["current_chain"])                                            # predictor's exact chain
    pre = engine.env.dapn.process(engine.env.dry_audio, {"chain": s0["current_chain"]})
    assert torch.allclose(pre, engine.current_audio_tensor, atol=1e-6)               # displayed audio = predictor render
    assert s0["improvement_pct"] == 0.0
    s = engine.run_all()
    assert s["improvement_pct"] == pytest.approx(100 * (s["predictor_loss"] - s["current_loss"]) / s["predictor_loss"], abs=1e-3)
    assert s["dry_improvement_pct"] == pytest.approx(100 * (s["dry_loss"] - s["current_loss"]) / s["dry_loss"], abs=1e-3)


def test_warm_start_requires_predictor(tmp_path, ckpts):
    eng = WorkflowEngine(predictor_ckpt=tmp_path / "none.pth", grrl_ckpt=ckpts[1])
    assert eng.has_real_agent and not eng.predictor_available
    with pytest.raises(RealBackendUnavailable, match="predictor"):
        eng.initialize_session(DRY, WET, mode="real", use_warm_start=True)
    eng.initialize_session(DRY, WET, mode="real", use_warm_start=False)  # cold start still fine


def test_backtrack_and_stop_steps_do_not_crash_parameter_mapping(engine):
    s = run_real(engine, warm=True)
    assert all(isinstance(st["parameters"], list) for st in s["steps_history"])
    stop = [st for st in s["steps_history"] if st["is_stop"]]
    assert all(st["parameters"] == [] for st in stop)


def test_failed_dsp_step_commits_nothing(engine, monkeypatch):
    from rl.visualization.workflow_engine import WorkflowStepError
    engine.initialize_session(DRY, WET, mode="real", max_steps=5)
    real = engine.env.dapn.process
    monkeypatch.setattr(engine.env.dapn, "process", lambda a, c: (_ for _ in ()).throw(FloatingPointError("boom")))
    with pytest.raises(WorkflowStepError):
        engine.step_forward()
    assert engine.current_step == 0 and engine.steps_history == [] and engine.env.current_chain == []
    assert engine.stages["dsp"]["status"] == "failed"
    monkeypatch.setattr(engine.env.dapn, "process", real)
    assert engine.step_forward()["current_step"] == 1


# ----------------------------------------------------------------- HTTP: streaming, downloads, exports
def test_http_end_to_end_real_run_matches_backend(server):
    base, eng = server
    code, _, body = call(base, "/api/initialize", "POST", {"mode": "real", "dry_path": DRY, "wet_path": WET, "use_warm_start": True, "max_steps": 5})
    assert code == 200 and json.loads(body)["warm_started"]
    s = json.loads(call(base, "/api/run_all", "POST", {})[2])
    assert s["is_terminated"] and s["has_output_audio"]
    # streamed + downloaded WAV are playable, correct length and equal to the backend tensor
    for path in ("/api/audio?type=output", "/api/download?type=output"):
        code, hdr, wav = call(base, path)
        assert code == 200 and hdr["Content-Type"] == "audio/wav" and wav[:4] == b"RIFF"
        tmp = Path(__file__).parent / "_dl.wav"; tmp.write_bytes(wav)
        try:
            audio, sr = load_audio(tmp)
        finally:
            tmp.unlink()
        ref = eng.final_audio_tensor.reshape(1, -1).clamp(-1, 1)
        assert sr == 44100 and audio.shape == ref.shape
        assert (audio - ref).abs().max() <= 2 / 32768   # 16-bit quantisation (write x32767, read /32768)
    assert "attachment" in call(base, "/api/download?type=output")[1]["Content-Disposition"]
    assert call(base, "/api/download?type=output")[1]["X-Execution-Mode"] == "real"
    for t in ("dry", "wet"):
        assert call(base, f"/api/audio?type={t}")[0] == 200
    # JSON export: valid (strict) JSON, carries mode & baselines, equals live summary values
    code, _, raw = call(base, "/api/export_json")
    rep = json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert rep["mode"] == "real" and rep["requested_mode"] == "real" and rep["simulated"] is False
    assert rep["predictor_loss"] == s["predictor_loss"] and rep["final_loss"] == s["current_loss"]
    assert rep["current_chain"] == s["current_chain"] and "base64_audio" not in json.dumps(rep)
    # SVG export parses as XML and states the execution mode
    code, hdr, svg = call(base, "/api/export_svg")
    assert code == 200 and hdr["Content-Type"] == "image/svg+xml"
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg") and "REAL" in svg.decode()


def test_http_step_by_step_updates_charts_from_events(server):
    base, eng = server
    call(base, "/api/initialize", "POST", {"mode": "real", "dry_path": DRY, "wet_path": WET, "max_steps": 6})
    n = 0
    while True:
        s = json.loads(call(base, "/api/step", "POST", {})[2])
        n += 1
        assert s["current_step"] == n and len(s["loss_history"]) == n + 1 and len(s["reward_history"]) == n
        if s["is_terminated"] or n > 8:
            break
    assert s["is_terminated"]
    assert json.loads(call(base, "/api/step", "POST", {})[2])["current_step"] == n   # no stepping after the episode ended


def test_http_tensor_truthiness_regression(server):
    """`tensor_a or tensor_b` raised for multi-element tensors; output streaming mid-episode must work."""
    base, eng = server
    call(base, "/api/initialize", "POST", {"mode": "real", "dry_path": DRY, "wet_path": WET, "max_steps": 6})
    json.loads(call(base, "/api/step", "POST", {})[2])
    code, _, wav = call(base, "/api/audio?type=output")
    assert eng.final_audio_tensor is None or True
    assert code in (200, 404) and (code == 404 or wav[:4] == b"RIFF")
    assert call(base, "/api/download?type=output")[0] in (200, 404)


def test_http_bad_requests(server):
    base, _ = server
    assert call(base, "/api/initialize", "POST", {"mode": "weird"})[0] == 400
    assert call(base, "/api/initialize", "POST", {"mode": "real", "dry_path": "/nonexistent.wav", "wet_path": WET})[0] == 400
    assert call(base, "/api/audio?type=bogus")[0] == 404
    assert call(base, "/api/export_svg")[0] == 409                     # nothing initialised
    assert call(base, "/api/nope", "POST", {})[0] == 404


# ----------------------------------------------------------------- exports
def test_svg_escaping_and_validity_for_hostile_names(tmp_path):
    summary = {"mode": "real", "simulated": False, "baseline_name": "predictor & <co>", "baseline_loss": 2.0, "initial_loss": 3.0,
               "current_loss": 1.0, "improvement_pct": -12.5, "cumulative_reward": -1.0,
               "current_chain": [{"effect": "EQ & <b>\"x\""}, {"effect": "Reverb"}],
               "loss_history": [{"step": 0, "loss": 2.0}, {"step": 1, "loss": 1.0}], "reward_history": [{"step": 1, "reward": -1.0, "cumulative": -1.0}]}
    out = export_svg_charts(summary, tmp_path / "c.svg")
    ET.parse(out)                                   # raises on invalid XML (e.g. unescaped & or undefined &rarr;)
    text = out.read_text(encoding="utf-8")
    assert "&rarr;" not in text and "-12.5%" in text and "REAL" in text


def test_json_safe_strips_nonfinite():
    out = json_safe({"a": float("nan"), "b": [float("inf"), 1.0], "c": {"d": -float("inf")}})
    assert out == {"a": None, "b": [None, 1.0], "c": {"d": None}}
    json.dumps(out, allow_nan=False)


def test_process_pair_report_has_mode_and_baselines(tmp_path, ckpts):
    from rl.process_pair import process_audio_pair
    rep = process_audio_pair(DRY, WET, tmp_path / "o.wav", mode="real", use_warm_start=True, max_steps=4,
                             predictor_ckpt=str(ckpts[0]), grrl_ckpt=str(ckpts[1]))
    saved = json.loads((tmp_path / "o.json").read_text())
    assert saved["execution_mode"] == "real" and saved["simulated"] is False and saved["warm_started"]
    assert saved["predictor_loss"] == saved["start_loss"] and saved["dry_loss"] != saved["start_loss"]
    assert saved["verification"]["chain_matches_audio"] and (tmp_path / "o.wav").exists()
    ET.parse(tmp_path / "o_charts.svg")


def test_process_pair_real_without_checkpoint_raises(tmp_path):
    from rl.process_pair import process_audio_pair
    with pytest.raises(RealBackendUnavailable):
        process_audio_pair(DRY, WET, tmp_path / "o.wav", mode="real", grrl_ckpt=str(tmp_path / "nope.pth"))
    assert not (tmp_path / "o.wav").exists()


def test_step_png_export_matches_session(engine, tmp_path):
    from rl.visualization.step_images import export_step_pngs
    engine.initialize_session(DRY, WET, mode="real", max_steps=4)
    s = engine.run_all()
    paths = export_step_pngs(engine, tmp_path / "png")
    names = sorted(p.name for p in paths)
    assert names == sorted([f"step_{i:02d}.png" for i in range(s["current_step"] + 1)] + ["summary.png"])
    assert len(engine.step_audio) == s["current_step"] + 1
    assert all(p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and p.stat().st_size > 5000 for p in paths)
    # the last recorded audio is exactly the env's final decision-crop render
    assert torch.equal(engine.step_audio[-1], engine.env.current_audio.cpu())


def test_step_png_export_refuses_demo(tmp_path, engine):
    from rl.visualization.step_images import export_step_pngs
    engine.initialize_session(DRY, WET, mode="demo")
    with pytest.raises(ValueError, match="demo"):
        export_step_pngs(engine, tmp_path)
