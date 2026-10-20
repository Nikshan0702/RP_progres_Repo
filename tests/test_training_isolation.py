"""Evaluation must not perturb training; scheduling, resume, leakage and mode-restoration guarantees."""
import copy
import json
import random
from pathlib import Path

import pytest
import torch

import rl.train_grrl as tg
from common.checkpoint import CheckpointError
from rl.evaluation import evaluate_grrl, grrl_episode, isolated_rng
from rl.grrl_agent import GRRLAgent
from rl.grrl_env import GRRLEnv

ROOT = Path(__file__).resolve().parent.parent


def pair(seed, n=22050):
    g = torch.Generator().manual_seed(seed)
    dry = torch.randn(1, n, generator=g) * .1
    return dry, torch.tanh(4 * dry)


def make_env(**kw):
    return GRRLEnv(max_chain_length=3, min_chain_length=1, allow_repeated_effects=False, max_episode_steps=6, **kw)


def env_state(env):
    return dict(chain=copy.deepcopy(env.current_chain), audio=env.current_audio.clone(), loss=env.current_loss,
                step=env.current_step, done=env._done, dry=env.dry_audio.clone(), target=env.target_audio.clone(),
                start=env.start_loss, mask=env.get_action_mask().clone())


def same(a, b):
    return all((torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k]) for k in a)


def test_evaluation_does_not_change_training_env_obs_rng_or_next_transition():
    def run(with_eval):
        torch.manual_seed(7); random.seed(7)
        train_env, eval_env = make_env(), make_env()
        agent = GRRLAgent(obs_dim=train_env.obs_dim, num_actions=train_env.num_actions)
        obs, _ = train_env.reset(*pair(1))
        for _ in range(2):
            m = train_env.get_action_mask(); a, c, _, _ = agent.select_action(obs, m)
            obs, *_ = train_env.step(a, c)
        snap, torch_rng, py_rng = env_state(train_env), torch.get_rng_state().clone(), random.getstate()
        if with_eval:
            agent.actor_critic.train()
            evaluate_grrl(eval_env, agent, [("a", *pair(2)), ("b", *pair(3))], warm_start=False, deterministic=False)  # stochastic = worst case
            assert agent.actor_critic.training, "model mode not restored"
            assert same(snap, env_state(train_env)) and torch.equal(torch_rng, torch.get_rng_state()) and py_rng == random.getstate()
        m = train_env.get_action_mask()
        a, c, lp, v = agent.select_action(obs, m)
        out = train_env.step(a, c)
        return a, c, lp, v, out[0], out[1], out[4]["current_loss"]
    A, B = run(False), run(True)
    assert A[0] == B[0] and torch.equal(A[1], B[1]) and A[2] == B[2] and A[3] == B[3]
    assert torch.equal(A[4], B[4]) and A[5] == B[5] and A[6] == B[6]


def test_model_mode_restored_in_both_directions():
    env = make_env(); agent = GRRLAgent(obs_dim=env.obs_dim, num_actions=env.num_actions)
    for mode in (True, False):
        agent.actor_critic.train(mode)
        evaluate_grrl(env, agent, [("a", *pair(1))], warm_start=False)
        assert agent.actor_critic.training is mode


def test_isolated_rng_restores_global_state():
    torch.manual_seed(0); random.seed(0)
    s = torch.get_rng_state().clone(); p = random.getstate()
    with isolated_rng():
        torch.rand(100); random.random()
    assert torch.equal(s, torch.get_rng_state()) and p == random.getstate()


def test_evaluation_records_errors_instead_of_hiding_them():
    env = make_env(); agent = GRRLAgent(obs_dim=env.obs_dim, num_actions=env.num_actions)
    bad = torch.full((1, 22050), float("nan"))
    recs = evaluate_grrl(env, agent, [("bad", bad, bad), ("ok", *pair(1))], warm_start=False)
    assert recs[0]["error"] and "bad" == recs[0]["sample_id"] and recs[1]["error"] is None


def tiny_metadata(tmp_path, n_train=6, n_val=4):
    def take(split, n):
        rows = [json.loads(l) for l in open(ROOT / f"dataset/synthetic/{split}/metadata.jsonl")][:n]
        p = tmp_path / f"{split}.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows)); return p
    return take("train", n_train), take("val", n_val)


def args_for(tmp_path, **over):
    tr, va = tiny_metadata(tmp_path)
    a = tg.build_parser().parse_args(["--total-timesteps", "96", "--rollout-steps", "32", "--eval-freq", "40", "--eval-episodes", "2",
                                      "--train-metadata", str(tr), "--val-metadata", str(va), "--checkpoint-dir", str(tmp_path / "ck"),
                                      "--max-chain-length", "3", "--max-episode-steps", "6"])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_eval_scheduling_uses_threshold_crossing_not_modulo(tmp_path):
    tg.train(args_for(tmp_path))
    steps = sorted(int(p.stem.split("_")[-1]) for p in (tmp_path / "ck" / "eval").glob("*.json"))
    # rollouts end at 32/64/96; eval_freq=40 never divides them, yet evaluations occur at 0,64,96
    assert steps == [0, 64, 96]


def test_validation_audio_never_reaches_training_env(tmp_path, monkeypatch):
    seen = {"train": set(), "eval": set()}
    real = GRRLEnv.reset
    def spy(self, dry, wet, *a, **k):
        seen["train" if self.failure_mode == "penalty" else "eval"].add(float(wet.double().sum()))
        return real(self, dry, wet, *a, **k)
    monkeypatch.setattr(GRRLEnv, "reset", spy)
    args = args_for(tmp_path); tg.train(args)
    assert seen["train"] and seen["eval"] and not (seen["train"] & seen["eval"])


def test_train_val_overlap_is_refused(tmp_path):
    tr, _ = tiny_metadata(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        tg.train(args_for(tmp_path, val_metadata=str(tr)))


def test_resume_missing_checkpoint_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        tg.train(args_for(tmp_path, resume=str(tmp_path / "nope.pth")))


def test_resume_restores_counters_optimizer_and_best_score(tmp_path):
    tg.train(args_for(tmp_path))
    ck = torch.load(tmp_path / "ck" / "last_grrl_model.pth", weights_only=False)
    assert ck["schema_version"] == 2 and ck["kind"] == "grrl" and ck["progress"]["timesteps"] == 96
    assert ck["optimizer_state"]["state"] and ck["rng_state"] and ck["provenance"]["datasets"]
    assert ck["selection"]["best_val_final_loss"] > 0
    agent = GRRLAgent()
    got = agent.load(tmp_path / "ck" / "last_grrl_model.pth", mode="resume")
    assert got["progress"]["updates"] == 3
    for k, v in ck["model_state"].items():
        assert torch.equal(agent.actor_critic.state_dict()[k], v)
    args = args_for(tmp_path, resume=str(tmp_path / "ck" / "last_grrl_model.pth"), total_timesteps=128)
    out = tg.train(args)
    assert out["steps"] == 128 and out["updates"] == 4


def test_eval_only_checkpoint_cannot_be_resumed(tmp_path):
    agent = GRRLAgent(); p = tmp_path / "w.pth"
    agent.save(p, include_optimizer=False)
    with pytest.raises(CheckpointError, match="cannot resume"):
        GRRLAgent().load(p, mode="resume")
    GRRLAgent().load(p, mode="eval")  # allowed


def test_missing_dataset_is_an_error_without_flag(tmp_path):
    with pytest.raises(SystemExit):
        tg.train(args_for(tmp_path, train_metadata=str(tmp_path / "missing.jsonl")))


def test_legacy_checkpoint_dir_is_protected():
    with pytest.raises(SystemExit):
        tg.train(tg.build_parser().parse_args(["--checkpoint-dir", str(ROOT / "checkpoints" / "grrl")]))


def test_truncation_is_bootstrapped_in_training_loop(tmp_path, monkeypatch):
    """Time-limit truncations must be stored with truncated=True and a next-state value."""
    from rl.grrl_buffer import GRRLRolloutBuffer
    seen = []
    real = GRRLRolloutBuffer.add
    def spy(self, *a, **k):
        seen.append((k.get("truncated", False), k.get("next_value"), a[7])); return real(self, *a, **k)
    monkeypatch.setattr(GRRLRolloutBuffer, "add", spy)
    tg.train(args_for(tmp_path, max_episode_steps=2))
    trunc = [s for s in seen if s[0]]
    assert trunc and all(nv is not None and done for _, nv, done in trunc)
