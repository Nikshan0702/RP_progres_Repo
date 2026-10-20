"""Optional in-place parameter edits (refinement without popping/re-adding)."""
import pytest
import torch

from rl.grrl_agent import GRRLAgent
from rl.grrl_env import BACKTRACK_ACTION_IDX, EFFECT_ACTIONS, GRRLEnv, STOP_ACTION_IDX

EQ, COMP = EFFECT_ACTIONS.index("EQ"), EFFECT_ACTIONS.index("Compressor")


def pair(seed=0, n=22050):
    g = torch.Generator().manual_seed(seed)
    dry = torch.randn(1, n, generator=g) * .1
    return dry, torch.tanh(3 * dry)


def make(edits=True, **kw):
    return GRRLEnv(max_chain_length=2, min_chain_length=1, allow_param_edits=edits, edit_range=.2, max_episode_steps=10, **kw)


def test_disabled_by_default_effect_in_chain_is_masked():
    env = make(edits=False); env.reset(*pair()); env.step(EQ)
    assert not env.get_action_mask()[EQ]
    with pytest.raises(ValueError):
        env.step(EQ)


def test_enabled_allows_editing_present_effects_even_at_max_length():
    env = make(); env.reset(*pair()); env.step(EQ, torch.tensor([.3, .6, .5])); env.step(COMP)
    m = env.get_action_mask()
    assert not m[EFFECT_ACTIONS.index("Delay")]       # cannot append beyond max length
    assert m[EQ] and m[COMP]                           # but may edit what is there


def test_edit_is_a_residual_in_place_and_keeps_length_and_position():
    env = make(); env.reset(*pair()); env.step(EQ, torch.tensor([.3, .6, .5])); env.step(COMP, torch.full((5,), .5))
    _, _, _, _, info = env.step(EQ, torch.tensor([.5, 1.0, 0.0]))     # a=.5 no change; a=1 -> +edit_range; a=0 -> -edit_range
    p = env.current_chain[0]["parameters"]
    assert info["is_edit"] and len(env.current_chain) == 2 and [s["effect"] for s in env.current_chain] == ["EQ", "Compressor"]
    assert p["frequency"] == pytest.approx(.3) and p["gain"] == pytest.approx(.8) and p["q"] == pytest.approx(.3)
    assert info["reward_breakdown"]["length_penalty"] == 0.0
    ref = env.dapn.process(env.dry_audio, {"chain": env.current_chain})
    assert torch.allclose(ref, env.current_audio, atol=1e-6)


def test_edit_clamps_to_unit_interval():
    env = make(); env.reset(*pair()); env.step(EQ, torch.tensor([.95, .05, .5]))
    env.step(EQ, torch.tensor([1., 0., .5]))
    p = env.current_chain[0]["parameters"]
    assert p["frequency"] == pytest.approx(1.0) and p["gain"] == pytest.approx(0.0)


def test_candidate_chain_is_pure_and_matches_step():
    env = make(); env.reset(*pair()); env.step(EQ, torch.tensor([.3, .6, .5]))
    before = [dict(s) for s in env.current_chain]
    cand, is_edit = env.candidate_chain(EQ, torch.tensor([.9, .5, .5]))
    assert is_edit and env.current_chain == before                      # no mutation
    env.step(EQ, torch.tensor([.9, .5, .5]))
    assert env.current_chain == cand


def test_backtrack_after_edit_and_stop_semantics_unchanged():
    env = make(); env.reset(*pair()); env.step(EQ); env.step(EQ, torch.full((3,), .7))
    env.step(BACKTRACK_ACTION_IDX)
    assert env.current_chain == []
    env.step(EQ); _, r, term, _, _ = env.step(STOP_ACTION_IDX)
    assert term and r == 0.0


def test_edits_and_repeats_are_mutually_exclusive():
    with pytest.raises(ValueError, match="ambiguous"):
        GRRLEnv(allow_param_edits=True, allow_repeated_effects=True)
    with pytest.raises(ValueError):
        GRRLEnv(edit_range=0.0)


def test_warm_start_refinement_by_edit(v2_predictor_ckpt):
    env = GRRLEnv(predictor_checkpoint=v2_predictor_ckpt, max_chain_length=4, allow_param_edits=True, max_episode_steps=8)
    _, info = env.reset(*pair(2), use_warm_start=True)
    first = env.current_chain[0]["effect"]
    _, _, _, _, i = env.step(EFFECT_ACTIONS.index(first), torch.full((5,), .6))
    assert i["is_edit"] and i["predictor_loss"] == info["predictor_loss"]       # baseline stays the predictor's
    assert [s["effect"] for s in env.current_chain][0] == first                  # order preserved


def test_training_records_edit_mode_and_resume_rejects_mismatch(tmp_path):
    import json
    from pathlib import Path
    import rl.train_grrl as tg
    root = Path(__file__).resolve().parent.parent
    def take(split, n):
        rows = [json.loads(l) for l in open(root / f"dataset/synthetic/{split}/metadata.jsonl")][:n]
        p = tmp_path / f"{split}.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows)); return str(p)
    base = ["--total-timesteps", "64", "--rollout-steps", "32", "--eval-freq", "32", "--eval-episodes", "2", "--max-chain-length", "3",
            "--max-episode-steps", "6", "--train-metadata", take("train", 4), "--val-metadata", take("val", 3), "--checkpoint-dir", str(tmp_path / "ck")]
    tg.train(tg.build_parser().parse_args(base + ["--allow-param-edits"]))
    ck = torch.load(tmp_path / "ck" / "last_grrl_model.pth", weights_only=False)
    assert ck["config"]["allow_param_edits"] is True and ck["config"]["edit_range"] == 0.25
    with pytest.raises(SystemExit, match="allow_param_edits"):
        tg.train(tg.build_parser().parse_args(base + ["--resume", str(tmp_path / "ck" / "last_grrl_model.pth"), "--total-timesteps", "96"]))


def test_dashboard_builds_env_like_training_and_shows_edits(tmp_path, v2_predictor_ckpt):
    from rl.visualization.config import DEFAULT_PRESETS
    from rl.visualization.workflow_engine import WorkflowEngine
    a = GRRLAgent(); a.save(tmp_path / "g.pth", extra_config={"allow_param_edits": True, "edit_range": 0.25, "warm_start": True})
    eng = WorkflowEngine(predictor_ckpt=v2_predictor_ckpt, grrl_ckpt=tmp_path / "g.pth")
    assert eng.has_real_agent and eng.env.allow_param_edits and eng.agent_config["allow_param_edits"] is True
    eng.initialize_session(DEFAULT_PRESETS[0]["dry_path"], DEFAULT_PRESETS[0]["wet_path"], mode="real", use_warm_start=True, max_steps=6)
    s = eng.run_all()
    for st in s["steps_history"]:
        if st.get("is_edit"):
            assert all(abs(p["normalized"] - p["prev_normalized"] - p["delta_normalized"]) < 1e-3 for p in st["parameters"] if p["prev_normalized"] is not None)
    assert s["verification"]["chain_matches_audio"]
