"""Predictor -> GRRL warm start: exact baseline, revisable chain, cold/warm separation."""
import pytest
import torch

from rl.grrl_env import BACKTRACK_ACTION_IDX, EFFECT_ACTIONS, GRRLEnv, STOP_ACTION_IDX


def pair(seed=0, n=22050):
    g = torch.Generator().manual_seed(seed)
    dry = torch.randn(1, n, generator=g) * .1
    return dry, torch.tanh(4 * dry)


@pytest.fixture
def env(v2_predictor_ckpt):
    return GRRLEnv(predictor_checkpoint=v2_predictor_ckpt, max_chain_length=5, min_chain_length=1,
                   allow_repeated_effects=False, max_episode_steps=12)


def test_initial_state_is_the_exact_predictor_chain(env):
    dry, wet = pair()
    _, info = env.reset(dry, wet, use_warm_start=True)
    d, w = env.dry_audio, env.target_audio
    pred = env.predictor.predict_chain(d, w, min_length=1, max_length=5)
    assert [s["effect"] for s in env.current_chain] == pred["sequence"]            # order = decoder order
    assert env.current_chain == pred["chain"]                                      # unrounded, identical parameters
    ref = env.loss_fn(env.dapn.process(d, {"chain": pred["chain"]}), w)
    assert info["predictor_loss"] == pytest.approx(float(ref), abs=1e-6) == pytest.approx(info["start_loss"], abs=1e-6)
    assert info["dry_loss"] == pytest.approx(float(env.loss_fn(d, w)), abs=1e-6)


def test_immediate_stop_reproduces_predictor_result_with_zero_reward(env):
    env.reset(*pair(), use_warm_start=True)
    audio0, loss0 = env.current_audio.clone(), env.current_loss
    _, r, term, trunc, info = env.step(STOP_ACTION_IDX)
    assert term and not trunc and r == 0.0 and info["current_loss"] == loss0
    assert torch.equal(env.current_audio, audio0)
    assert info["predictor_loss"] == info["start_loss"] == loss0 and info["improvement_pct"] == 0.0


def test_warm_chain_can_be_revised_by_backtracking_and_reordering(env):
    env.reset(*pair(1), use_warm_start=True)
    original = [s["effect"] for s in env.current_chain]
    n = len(original)
    # pop the whole predicted prefix (only legal while the minimum chain length stays reachable)
    steps = 0
    while len(env.current_chain) > 0 and env.get_action_mask()[BACKTRACK_ACTION_IDX]:
        env.step(BACKTRACK_ACTION_IDX); steps += 1
    assert steps >= 1 and len(env.current_chain) < n          # the predicted prefix really was removed
    # re-add effects in the REVERSE order -> a different chain (revision, not append-only)
    for name in reversed(original):
        idx = EFFECT_ACTIONS.index(name)
        if env.get_action_mask()[idx] and env.current_step < env.max_episode_steps:
            env.step(idx, torch.full((5,), .5))
    ref = env.dapn.process(env.dry_audio, {"chain": env.current_chain})
    assert torch.allclose(ref, env.current_audio, atol=1e-6)
    assert env.predictor_loss is not None                          # baseline still the predictor's, not the revised chain's


def test_predictor_baseline_is_constant_over_the_episode_and_improvement_is_signed(env):
    _, info0 = env.reset(*pair(2), use_warm_start=True)
    base = info0["predictor_loss"]
    seen = []
    done = False
    while not done:
        mask = env.get_action_mask()
        a = int(torch.nonzero(mask)[0])                       # first valid action: arbitrary, deterministic
        _, _, term, trunc, info = env.step(a, torch.full((5,), .5))
        seen.append(info)
        done = term or trunc
    assert all(i["predictor_loss"] == base for i in seen)
    for i in seen:
        assert i["improvement_pct"] == pytest.approx(100 * (base - i["current_loss"]) / base, abs=1e-6)  # may be negative


def test_cold_and_warm_runs_are_kept_separate(env):
    dry, wet = pair(3)
    _, w = env.reset(dry, wet, use_warm_start=True)
    _, c = env.reset(dry, wet, use_warm_start=False)
    assert w["warm_started"] and w["predictor_loss"] is not None and w["chain_length"] >= 1
    assert not c["warm_started"] and c["predictor_loss"] is None and c["chain_length"] == 0
    assert c["start_loss"] == c["dry_loss"] and env.current_chain == []


def test_warm_start_respects_configured_chain_limits(v2_predictor_ckpt):
    e = GRRLEnv(predictor_checkpoint=v2_predictor_ckpt, max_chain_length=2, min_chain_length=2, max_episode_steps=6)
    for s in range(4):
        e.reset(*pair(s), use_warm_start=True)
        assert 2 <= len(e.current_chain) <= 2
