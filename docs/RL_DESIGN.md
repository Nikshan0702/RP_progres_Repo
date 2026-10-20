# GRRL / PPO design contract (what the code guarantees)

## Distinct quantities (never interchangeable)
| name | where | definition | direction |
|---|---|---|---|
| audio reconstruction loss | `common/losses.py` (`MultiResolutionSTFTLoss`) | spectral-convergence + log-magnitude STFT distance between rendered and target audio | **lower = better** |
| environment reward | `GRRLEnv.step` | see components below | higher = better |
| predictor supervised loss | `train.py` | presence BCE + 2·param MSE + order CE | lower = better |
| PPO optimisation loss | `GRRLAgent.train_step` | clipped surrogate + c₁·value MSE − c₂·entropy | lower = better |

## Reward components (per step; sum is asserted equal to `info['reward']`)
* `loss_improvement_reward = +10·(prev_loss − new_loss)` (positive when the loss drops; negative = degradation)
* `step_penalty = −step_penalty` for every operation except STOP
* `length_penalty = −length_penalty·(len_after − len_before)` (BACKTRACK earns the credit back)
* `terminal_bonus = +3` when the target loss is reached and the episode started above target
* render failure (training env, `failure_mode='penalty'`): `failure_penalty = −1`, nothing committed, budget consumed

## Actions and chain validity (enforced in the environment, not only in the policy mask)
`0..6` append an effect (params ∈ (0,1)), `7` STOP, `8` BACKTRACK (remove last effect).
Masked/invalid actions raise `ValueError` **before** any state change; stepping a finished episode raises
`RuntimeError`; min/max chain length and the no-repeat rule come from `config/system_config.yaml` (shared with DAPN).
STOP is masked until `min_chain_length` is met. BACKTRACK is masked when it would make the minimum length unreachable within the budget.
Termination: STOP or target reached (**terminal**, bootstrap 0). Budget exhaustion is **truncation** (bootstrap V(s_next)).

## Bounded continuous policy
Beta(α, β) on (0,1), α = 1 + m·k, β = 1 + (1−m)·k, k = exp(−2·log_scale), m = sigmoid(head). Shapes ≥ 1 ⇒ finite density
everywhere; samples are clamped to [1e-6, 1−1e-6] **before** being stored/evaluated/sent to the environment, so PPO's
old and new log-probabilities describe the stored action. Entropy is the analytic Beta differential entropy (nats, ≤ 0).
Deterministic evaluation uses the mode (= m). Physical values are the affine `ParameterMapper` image of the action.

## Warm start
`reset(use_warm_start=True)` renders the inverse predictor's decoded chain (ordered, constrained to `[min,max]` length)
and uses its **actual** loss as `predictor_loss`/`start_loss`. The agent revises the chain with BACKTRACK + re-append (and
optional parameter edits) or stops. Immediate STOP reproduces the predictor result with reward exactly 0.
Cold start begins from an empty chain; both modes are evaluated separately.

## Training / evaluation separation
Two `GRRLEnv`/`AudioFXEnv` instances; evaluation runs under `isolated_rng()` and restores the policy's train/eval mode;
evaluation is scheduled by threshold crossing; checkpoint selection uses a track-disjoint `valsel` split and **final claims use
a different `test` split** (`dataset/make_splits.py`, `rl/evaluate_protocol.py`).

## Checkpoints (schema v2, `common/checkpoint.py`)
weights, optimizer, scheduler, python/numpy/torch RNG, progress counters, sampler state, architecture + distribution config,
best-validation metadata, dataset hashes, library versions, git commit. Missing resume file → `FileNotFoundError`;
legacy/foreign/incompatible files → `IncompatibleCheckpointError`; eval-only files cannot be resumed. Legacy checkpoints in
`checkpoints/` are never overwritten (a backup lives outside the repo).

## Audio limits
44.1 kHz mono everywhere (`audio_io.load_audio` resamples/downmixes; tensors with another rate are rejected by
`common.audio_pair.prepare_audio_pair`). Two limits: the **10 s system limit** (`audio.max_duration_seconds`, longer files are
truncated with a logged warning) and the **2 s RL/predictor crop** (first 2 s of the aligned pair; rewards, losses and the
dashboard all use the cropped pair). Inputs shorter than 1025 samples are zero-padded so STFT front-ends cannot fail.
