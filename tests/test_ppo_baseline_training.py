"""Fixed-slot PPO baseline: training harness, evaluation isolation, checkpoints."""
import json
import random
from pathlib import Path

import pytest
import torch

import rl.train_ppo as tp
from common.checkpoint import IncompatibleCheckpointError
from rl.dapn_env import AudioFXEnv
from rl.evaluation import evaluate_ppo
from rl.ppo_agent import PPOAgent

ROOT = Path(__file__).resolve().parent.parent


def pair(seed, n=22050):
    g = torch.Generator().manual_seed(seed)
    dry = torch.randn(1, n, generator=g) * .1
    return dry, torch.tanh(4 * dry)


def test_ppo_eval_leaves_training_env_and_rng_untouched():
    torch.manual_seed(0); random.seed(0)
    tr, ev = AudioFXEnv(max_steps_per_episode=4), AudioFXEnv(max_steps_per_episode=4)
    agent = PPOAgent(obs_dim=tr.obs_dim, discrete_dim=tr.num_effects, continuous_dim=tr.num_params)
    obs, _ = tr.reset(*pair(1)); a = agent.select_action(obs)[0]; obs, *_ = tr.step(a)
    snap = (tr.current_audio.clone(), tr.current_loss, tr.current_step, tr.current_presence.clone(), tr.current_params.clone())
    rng = torch.get_rng_state().clone(); agent.actor_critic.train()
    evaluate_ppo(ev, agent, [("a", *pair(2))], warm_start=False, deterministic=False)
    assert agent.actor_critic.training
    assert torch.equal(snap[0], tr.current_audio) and snap[1:3] == (tr.current_loss, tr.current_step)
    assert torch.equal(snap[3], tr.current_presence) and torch.equal(snap[4], tr.current_params) and torch.equal(rng, torch.get_rng_state())


def test_ppo_training_smoke_checkpoint_and_resume(tmp_path):
    def take(split, n):
        rows = [json.loads(l) for l in open(ROOT / f"dataset/synthetic/{split}/metadata.jsonl")][:n]
        p = tmp_path / f"{split}.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows)); return str(p)
    args = tp.build_parser().parse_args(["--total-timesteps", "64", "--rollout-steps", "32", "--eval-freq", "20", "--eval-episodes", "2",
                                         "--train-metadata", take("train", 5), "--val-metadata", take("val", 3),
                                         "--checkpoint-dir", str(tmp_path / "ppo"), "--max-episode-steps", "4"])
    out = tp.train(args)
    assert out["steps"] == 64 and out["updates"] == 2
    ck = torch.load(tmp_path / "ppo" / "last_ppo_model.pth", weights_only=False)
    assert ck["kind"] == "ppo_fixed_slot" and ck["optimizer_state"] and ck["progress"]["timesteps"] == 64
    args.resume = str(tmp_path / "ppo" / "last_ppo_model.pth"); args.total_timesteps = 96
    assert tp.train(args)["steps"] == 96
    args.resume = str(tmp_path / "missing.pth")
    with pytest.raises(FileNotFoundError):
        tp.train(args)


def test_legacy_ppo_checkpoint_rejected():
    legacy = ROOT / "checkpoints/ppo/best_ppo_model.pth"
    if not legacy.exists():
        pytest.skip("legacy checkpoint absent")
    with pytest.raises(IncompatibleCheckpointError, match="legacy"):
        PPOAgent().load(legacy)
