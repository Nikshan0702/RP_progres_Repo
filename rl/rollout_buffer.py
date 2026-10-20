"""
rl/rollout_buffer.py
====================
Rollout buffer and Generalized Advantage Estimation (GAE) for PPO training.

Stores transitions collected from environment interaction, computes GAE advantages
and discounted returns, and provides shuffled mini-batch iterators for policy updates.
"""

from __future__ import annotations

from typing import Generator
import torch


class RolloutBuffer:
    """Storage buffer for on-policy PPO rollouts.

    Parameters
    ----------
    buffer_size : int
        Number of steps to store in the buffer.
    obs_dim : int
        Observation vector dimensionality (default: 100).
    discrete_dim : int
        Number of discrete presence controls (default: 7).
    continuous_dim : int
        Number of continuous parameter controls (default: 29).
    device : torch.device | str
        Target device for stored tensors.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int = 100,
        discrete_dim: int = 7,
        continuous_dim: int = 29,
        device: torch.device | str = "cpu",
    ) -> None:
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim
        self.device = torch.device(device)

        self.reset()

    def reset(self) -> None:
        """Clear the buffer and reset pointer."""
        self.observations = torch.zeros((self.buffer_size, self.obs_dim), dtype=torch.float32, device=self.device)
        self.discrete_actions = torch.zeros((self.buffer_size, self.discrete_dim), dtype=torch.float32, device=self.device)
        self.continuous_actions = torch.zeros((self.buffer_size, self.continuous_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)

        self.truncated = torch.zeros(self.buffer_size, dtype=torch.bool, device=self.device)
        self.timeout_values = torch.zeros(self.buffer_size, device=self.device)

        self.advantages = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)

        self.ptr = 0
        self.is_full = False

    def add(
        self,
        obs: torch.Tensor,
        discrete_action: torch.Tensor,
        continuous_action: torch.Tensor,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        truncated: bool = False,
        next_value: float | None = None,
    ) -> None:
        """Insert a transition step into the buffer."""
        if self.ptr >= self.buffer_size:
            raise IndexError("RolloutBuffer overflow: attempt to add transition beyond buffer_size.")

        self.observations[self.ptr] = obs.to(self.device)
        self.discrete_actions[self.ptr] = discrete_action.to(self.device)
        self.continuous_actions[self.ptr] = continuous_action.to(self.device)
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        if truncated and (not done or next_value is None):
            raise ValueError("Truncation requires done=True and terminal observation value")
        self.dones[self.ptr] = 1.0 if done else 0.0
        self.truncated[self.ptr] = truncated
        self.timeout_values[self.ptr] = 0.0 if next_value is None else next_value

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.is_full = True

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE) and discounted return targets.

        Parameters
        ----------
        last_value : float
            Value prediction for state s_{T} (bootstrap value).
        last_done : bool
            Whether the final state was terminal.
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE decay parameter.
        """
        last_val = float(last_value)
        last_d = 1.0 if last_done else 0.0
        gae = 0.0

        for t in reversed(range(self.ptr)):
            boundary = bool(self.dones[t]) or (t == self.ptr-1 and last_done)
            timeout = bool(self.truncated[t])
            if timeout:
                next_val = float(self.timeout_values[t])
            elif boundary:
                next_val = 0.0
            else:
                next_val = last_val if t == self.ptr-1 else float(self.values[t+1])
            delta = float(self.rewards[t]) + gamma * next_val - float(self.values[t])
            # Bootstrap time limits, but never propagate GAE into a reset episode.
            gae = delta + gamma * gae_lambda * (not boundary) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_minibatches(
        self,
        batch_size: int,
        normalize_advantages: bool = True,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield randomized mini-batches for PPO parameter updates."""
        total_steps = self.ptr
        if total_steps == 0:
            return

        advantages = self.advantages[:total_steps]
        if normalize_advantages and total_steps > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            norm_advantages = (advantages - adv_mean) / adv_std
        else:
            norm_advantages = advantages

        indices = torch.randperm(total_steps, device=self.device)

        for start_idx in range(0, total_steps, batch_size):
            end_idx = min(start_idx + batch_size, total_steps)
            mb_indices = indices[start_idx:end_idx]

            yield {
                "observations": self.observations[mb_indices],
                "discrete_actions": self.discrete_actions[mb_indices],
                "continuous_actions": self.continuous_actions[mb_indices],
                "old_values": self.values[mb_indices],
                "old_log_probs": self.log_probs[mb_indices],
                "advantages": norm_advantages[mb_indices],
                "returns": self.returns[mb_indices],
            }
