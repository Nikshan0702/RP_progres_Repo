"""GAE correctness (terminal vs truncation), PPO ratio/clipping, and verified learning on toy problems."""
import math
import pytest
import torch

from rl.grrl_agent import GRRLAgent, clamp_actions, bounded_parameters
from rl.grrl_buffer import GRRLRolloutBuffer

OBS, NA = 188, 9


def fill(buf, rewards, values, dones, truncs=None, next_values=None):
    truncs = truncs or [False] * len(rewards)
    for t, (r, v, d) in enumerate(zip(rewards, values, dones)):
        buf.add(torch.zeros(OBS), torch.ones(NA, dtype=torch.bool), 0, torch.zeros(5), r, v, 0.0, d,
                truncated=truncs[t], next_value=(next_values or {}).get(t))


def reference_gae(rewards, values, dones, truncs, next_values, last_value, last_done, g, lam):
    """Independent O(n^2) definition: A_t = sum_k (g*lam)^k delta_{t+k} until an episode boundary."""
    n = len(rewards); adv = []
    for t in range(n):
        total, k = 0.0, 0
        while t + k < n:
            i = t + k
            boundary = dones[i] or (i == n - 1 and last_done)
            if truncs[i]:
                nxt = next_values[i]
            elif boundary:
                nxt = 0.0
            else:
                nxt = last_value if i == n - 1 else values[i + 1]
            total += (g * lam) ** k * (rewards[i] + g * nxt - values[i])
            if boundary:
                break
            k += 1
        adv.append(total)
    return adv


def test_terminal_vs_truncated_bootstrap():
    b = GRRLRolloutBuffer(1, OBS, NA); fill(b, [1.0], [0.5], [True])
    b.compute_returns_and_advantages(0.0, True, gamma=.9, gae_lambda=.95)
    assert b.advantages[0].item() == pytest.approx(1.0 - 0.5)                      # terminal: no bootstrap
    b = GRRLRolloutBuffer(1, OBS, NA); fill(b, [1.0], [0.5], [True], [True], {0: 2.0})
    b.compute_returns_and_advantages(0.0, True, gamma=.9, gae_lambda=.95)
    assert b.advantages[0].item() == pytest.approx(1.0 + .9 * 2.0 - 0.5)           # timeout: bootstraps V(s_next)
    assert b.returns[0].item() == pytest.approx(1.0 + .9 * 2.0)


def test_truncation_requires_done_and_value():
    b = GRRLRolloutBuffer(2, OBS, NA)
    with pytest.raises(ValueError):
        fill(b, [1.], [0.], [False], [True], {0: 1.0})
    with pytest.raises(ValueError):
        fill(b, [1.], [0.], [True], [True], {})


@pytest.mark.parametrize("seed", range(5))
def test_gae_matches_reference_with_mixed_episodes(seed):
    g = torch.Generator().manual_seed(seed)
    n = 12
    rewards = torch.randn(n, generator=g).tolist(); values = torch.randn(n, generator=g).tolist()
    dones = [False] * n; truncs = [False] * n; nv = {}
    for i in (3, 7):
        dones[i] = True
    truncs[7] = True; nv[7] = 1.7
    last_done = seed % 2 == 0
    if last_done:
        dones[-1] = True
    b = GRRLRolloutBuffer(n, OBS, NA); fill(b, rewards, values, dones, truncs, nv)
    b.compute_returns_and_advantages(last_value=0.3, last_done=last_done, gamma=.97, gae_lambda=.9)
    ref = reference_gae(rewards, values, dones, truncs, nv, 0.3, last_done, .97, .9)
    assert torch.allclose(b.advantages, torch.tensor(ref), atol=1e-5)


def test_advantage_does_not_leak_across_reset():
    b = GRRLRolloutBuffer(4, OBS, NA)
    fill(b, [0., 0., 100., 0.], [0., 0., 0., 0.], [False, True, False, False])
    b.compute_returns_and_advantages(0.0, False, gamma=1.0, gae_lambda=1.0)
    assert b.advantages[1].item() == 0.0 and b.advantages[0].item() == 0.0   # reward after the reset cannot reach t<=1


def test_ratio_is_one_for_unchanged_policy_and_clip_zeroes_gradient():
    torch.manual_seed(0)
    a = GRRLAgent(); obs = torch.randn(6, OBS); masks = torch.ones(6, NA, dtype=torch.bool)
    acts = torch.tensor([0, 1, 2, 3, 4, 5]); cont = torch.rand(6, 5)
    lp, ent, _ = a.actor_critic.evaluate_actions(obs, masks, acts, cont)
    lp2, _, _ = a.actor_critic.evaluate_actions(obs, masks, acts, cont)
    assert torch.allclose((lp2 - lp.detach()).exp(), torch.ones(6), atol=1e-6)
    # ratio outside the clip range with a positive advantage => zero policy gradient (min picks the clipped term)
    ratio = torch.tensor([1.5], requires_grad=True); adv = torch.tensor([1.0])
    obj = torch.min(ratio * adv, ratio.clamp(.8, 1.2) * adv); obj.sum().backward()
    assert ratio.grad.item() == 0.0
    ratio = torch.tensor([.5], requires_grad=True)  # negative advantage, ratio below range => also clipped
    obj = torch.min(ratio * -adv, ratio.clamp(.8, 1.2) * -adv); obj.sum().backward()
    assert ratio.grad.item() == 0.0


def test_boundary_actions_are_finite_after_clamp():
    d = bounded_parameters(torch.tensor([.1, .9, .5]), torch.zeros(3))
    x = clamp_actions(torch.tensor([0., 1., .5]))
    assert torch.isfinite(d.log_prob(x)).all() and ((x > 0) & (x < 1)).all()


def make_buffer(agent, obs, n, reward_fn):
    buf = GRRLRolloutBuffer(n, OBS, NA)
    mask = torch.zeros(NA, dtype=torch.bool); mask[:2] = True
    for _ in range(n):
        a, c, lp, v = agent.select_action(obs, mask)
        buf.add(obs, mask, a, c, reward_fn(a, c), v, lp, True)
    buf.compute_returns_and_advantages(0.0, True, gamma=.99, gae_lambda=.95)
    return buf


def test_ppo_improves_expected_reward_on_discrete_and_continuous_bandit():
    """Evidence of LEARNING (expected return rises), not merely of weight change."""
    torch.manual_seed(3)
    agent = GRRLAgent(lr=3e-3, entropy_coef=0.0); obs = torch.randn(OBS)
    reward = lambda a, c: (1.0 if a == 0 else -1.0) - (float(c[0]) - .8) ** 2 * 10 * (a == 0)
    def expected(n=300):
        return sum(reward(*agent.select_action(obs, torch.tensor([True, True] + [False] * 7))[:2]) for _ in range(n)) / n
    before = expected()
    w0 = [p.detach().clone() for p in agent.actor_critic.parameters()]
    for _ in range(25):
        buf = make_buffer(agent, obs, 64, reward)
        m = agent.train_step(buf, batch_size=32, epochs=4)
        assert all(math.isfinite(v) for v in m.values())
    after = expected()
    assert any(not torch.equal(a, b) for a, b in zip(w0, agent.actor_critic.parameters()))
    assert after > before + 0.8, (before, after)
    # continuous head moved toward the optimum 0.8 for effect 0 (EQ.frequency)
    feats = agent.actor_critic.forward_features(obs[None])[0]
    assert abs(float(torch.sigmoid(agent.actor_critic.param_heads["EQ"](feats))[0]) - .8) < .25


def test_gradients_reach_exactly_the_used_heads_and_are_finite():
    torch.manual_seed(1)
    agent = GRRLAgent(); obs = torch.randn(8, OBS); masks = torch.ones(8, NA, dtype=torch.bool)
    acts = torch.zeros(8, dtype=torch.long)                       # only EQ chosen
    lp, ent, v = agent.actor_critic.evaluate_actions(obs, masks, acts, torch.rand(8, 5))
    (-(lp).mean() + v.pow(2).mean() - ent.mean()).backward()
    ac = agent.actor_critic
    for name, p in ac.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name
    assert ac.param_heads["EQ"].weight.grad.abs().sum() > 0 and ac.param_log_stds["EQ"].grad.abs().sum() > 0
    assert ac.param_heads["Reverb"].weight.grad is None or ac.param_heads["Reverb"].weight.grad.abs().sum() == 0
    assert ac.discrete_head.weight.grad.abs().sum() > 0 and ac.value_head.weight.grad.abs().sum() > 0


def test_old_log_prob_in_buffer_equals_current_policy_log_prob_for_stored_action():
    torch.manual_seed(5); agent = GRRLAgent(); obs = torch.randn(OBS)
    buf = make_buffer(agent, obs, 16, lambda a, c: 0.0)
    batch = next(buf.get_minibatches(16))
    lp, _, _ = agent.actor_critic.evaluate_actions(batch["observations"], batch["action_masks"], batch["discrete_actions"], batch["continuous_actions"])
    assert torch.allclose(lp, batch["old_log_probs"], atol=1e-4)
