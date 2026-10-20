"""Environment action contract, termination semantics, rendering failures, reward reconciliation."""
import math
import pytest
import torch

from rl.grrl_env import BACKTRACK_ACTION_IDX, EFFECT_ACTIONS, GRRLEnv, STOP_ACTION_IDX

EQ, COMP, DELAY, REVERB, DIST = (EFFECT_ACTIONS.index(n) for n in ["EQ", "Compressor", "Delay", "Reverb", "Distortion"])


@pytest.fixture
def pair():
    torch.manual_seed(0)
    t = torch.arange(22050) / 44100
    dry = (0.4 * torch.sin(2 * math.pi * 220 * t) + 0.05 * torch.randn(22050))[None]
    return dry, torch.tanh(3 * dry)


def make(**kw):
    kw.setdefault("max_chain_length", 3); kw.setdefault("min_chain_length", 1)
    kw.setdefault("allow_repeated_effects", False); kw.setdefault("max_episode_steps", 6)
    return GRRLEnv(**kw)


def test_stop_masked_and_rejected_before_min_length(pair):
    env = make(min_chain_length=2); env.reset(*pair)
    assert not env.get_action_mask()[STOP_ACTION_IDX]
    with pytest.raises(ValueError, match="masked"):
        env.step(STOP_ACTION_IDX)
    assert env.current_chain == [] and env.current_step == 0  # nothing mutated
    env.step(EQ)
    with pytest.raises(ValueError):
        env.step(STOP_ACTION_IDX)
    env.step(COMP)
    assert env.get_action_mask()[STOP_ACTION_IDX]


def test_repeat_rejected_without_appending(pair):
    env = make(); env.reset(*pair); env.step(EQ)
    with pytest.raises(ValueError, match="masked"):
        env.step(EQ)
    assert [s["effect"] for s in env.current_chain] == ["EQ"] and env.current_step == 1


def test_repeat_allowed_when_configured(pair):
    env = make(allow_repeated_effects=True); env.reset(*pair); env.step(EQ); env.step(EQ)
    assert len(env.current_chain) == 2


def test_max_length_blocks_more_effects_and_cannot_exceed(pair):
    env = make(max_chain_length=2); env.reset(*pair); env.step(EQ); env.step(COMP)
    assert not env.get_action_mask()[:7].any()
    with pytest.raises(ValueError):
        env.step(DELAY)
    assert len(env.current_chain) == 2


@pytest.mark.parametrize("bad", [-1, 9, 100, 1.5, float("nan"), [1, 2]])
def test_invalid_discrete_actions_raise_without_mutation(pair, bad):
    env = make(); env.reset(*pair)
    with pytest.raises((ValueError, RuntimeError)):
        env.step(bad)
    assert env.current_step == 0 and env.current_chain == []


@pytest.mark.parametrize("params", [[1.5, .5, .5], [-0.1, .5, .5], [float("nan"), .5, .5], [.5, .5], [[.5, .5, .5]]])
def test_invalid_parameters_rejected(pair, params):
    env = make(); env.reset(*pair)
    with pytest.raises(ValueError):
        env.step(EQ, torch.tensor(params))
    assert env.current_chain == [] and env.current_step == 0


def test_no_step_after_termination_or_truncation(pair):
    env = make(); env.reset(*pair); env.step(EQ)
    _, _, terminated, truncated, _ = env.step(STOP_ACTION_IDX)
    assert terminated and not truncated
    with pytest.raises(RuntimeError, match="finished"):
        env.step(COMP)
    assert not env.get_action_mask().any()
    env = make(max_episode_steps=2, target_threshold_loss=-1.); env.reset(*pair)
    env.step(EQ); _, _, terminated, truncated, info = env.step(COMP)
    assert truncated and not terminated and info["truncated"] and not info["terminated"]
    with pytest.raises(RuntimeError):
        env.step(DELAY)


def test_truncation_is_not_termination_and_stop_is_terminal(pair):
    env = make(max_episode_steps=1, target_threshold_loss=-1.); env.reset(*pair)
    _, _, term, trunc, _ = env.step(EQ)
    assert (term, trunc) == (False, True)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        make().step(0)


def test_backtrack_removes_last_effect_and_restores_audio(pair):
    env = make(); env.reset(*pair)
    assert not env.get_action_mask()[BACKTRACK_ACTION_IDX]
    env.step(EQ); loss_after_eq = env.current_loss
    env.step(COMP)
    env.step(BACKTRACK_ACTION_IDX)
    assert [s["effect"] for s in env.current_chain] == ["EQ"]
    assert abs(env.current_loss - loss_after_eq) < 1e-6
    ref = env.dapn.process(env.dry_audio, {"chain": env.current_chain})
    assert torch.allclose(ref, env.current_audio, atol=1e-6)


def test_reported_chain_matches_audio_after_every_step(pair):
    env = make(max_episode_steps=8); env.reset(*pair)
    for a in [EQ, COMP, BACKTRACK_ACTION_IDX, REVERB]:
        env.step(a, torch.rand(5))
        ref = env.dapn.process(env.dry_audio, {"chain": env.current_chain})
        assert torch.allclose(ref, env.current_audio, atol=1e-6)
        assert abs(float(env.loss_fn(ref, env.target_audio)) - env.current_loss) < 1e-6


def _failing_dapn(env, fail_on):
    real = env.dapn.process
    def process(audio, chain):
        if any(s["effect"] == fail_on for s in chain["chain"]):
            raise FloatingPointError("injected DSP failure")
        return real(audio, chain)
    env.dapn.process = process


def test_render_failure_raise_mode_is_transactional(pair):
    env = make(); env.reset(*pair); env.step(EQ)
    before = (list(env.current_chain), env.current_audio.clone(), env.current_loss, env.current_step)
    _failing_dapn(env, "Reverb")
    with pytest.raises(RuntimeError, match="injected"):
        env.step(REVERB)
    assert env.current_chain == before[0] and torch.equal(env.current_audio, before[1])
    assert env.current_loss == before[2] and env.current_step == before[3]


def test_render_failure_penalty_mode_is_explicit_and_consistent(pair):
    env = make(failure_mode="penalty", failure_penalty=2.0); env.reset(*pair); env.step(EQ)
    audio, loss = env.current_audio.clone(), env.current_loss
    _failing_dapn(env, "Reverb")
    obs, reward, term, trunc, info = env.step(REVERB)
    assert info["render_failed"] and "injected" in info["error"]
    assert reward == -2.0 == info["reward_breakdown"]["total_step_reward"] and not term
    assert [s["effect"] for s in env.current_chain] == ["EQ"]  # failed effect is NOT in the reported chain
    assert torch.equal(env.current_audio, audio) and env.current_loss == loss
    assert env.current_step == 2  # failure consumed budget -> cannot loop forever


def test_nonfinite_render_is_a_failure_not_dry_fallback(pair):
    env = make(failure_mode="penalty"); env.reset(*pair)
    real = env.dapn.process
    env.dapn.process = lambda a, c: real(a, c) * float("nan")
    _, r, _, _, info = env.step(EQ)
    assert info["render_failed"] and env.current_chain == []


def test_reward_reconciles_and_has_documented_signs(pair):
    env = make(length_penalty=.02, step_penalty=.05, target_threshold_loss=-1.); env.reset(*pair)
    for a in [EQ, COMP, BACKTRACK_ACTION_IDX, STOP_ACTION_IDX]:
        prev = env.current_loss; n0 = len(env.current_chain)
        _, reward, _, _, info = env.step(a, torch.rand(5))
        b = info["reward_breakdown"]
        assert abs(sum(v for k, v in b.items() if k != "total_step_reward") - reward) < 1e-9
        assert abs(b["total_step_reward"] - reward) < 1e-9
        assert abs(b["loss_improvement_reward"] - 10 * (prev - info["current_loss"])) < 1e-9  # lower loss => positive
        assert abs(b["length_penalty"] + .02 * (info["chain_length"] - n0)) < 1e-9
        assert b["step_penalty"] == (0. if a == STOP_ACTION_IDX else -.05)
        assert info["delta_loss"] == pytest.approx(prev - info["current_loss"])


def test_target_bonus_only_when_started_above_target(pair):
    env = make(target_threshold_loss=1e9); env.reset(*pair)  # already below target at start
    _, _, term, _, info = env.step(EQ)
    assert term and info["reward_breakdown"]["terminal_bonus"] == 0.
    env = make(target_threshold_loss=1e-9); env.reset(*pair)
    assert env.start_loss > 1e-9


def test_improvement_percent_reports_degradation(pair):
    from rl.grrl_env import improvement_percent
    assert improvement_percent(1.0, 1.5) == pytest.approx(-50.0)  # not clamped to zero
    assert improvement_percent(1.0, .25) == pytest.approx(75.0)


def test_env_and_dapn_configuration_agree():
    from common.config_loader import load_system_config, get_sample_rate
    env = GRRLEnv()
    cfg = load_system_config()["chain"]
    assert env.max_chain_length == cfg["max_chain_length"] and env.min_chain_length == cfg["min_chain_length"]
    assert env.allow_repeated_effects == cfg["allow_repeated_effects"]
    assert env.sample_rate == get_sample_rate() == env.dapn._sample_rate
    with pytest.raises(ValueError):
        GRRLEnv(sample_rate=48000)
    assert env.audio_duration_sec == 2.0 and env.max_samples == 88200
