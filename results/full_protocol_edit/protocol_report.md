# Held-out evaluation (n=248, dataset/synthetic_v2/test/metadata.jsonl)

| method | mean loss | std | median | vs dry: mean Δ [95% CI] | improved/degraded % | vs predictor: mean Δ [95% CI] | improved/degraded % | stop% | steps | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| dry | 2.1935 | 1.9009 | 1.6164 | - | - | - | - | nan | nan | 0 |
| predictor_only | 1.3577 | 0.7334 | 1.3338 | +0.8358 [+0.6471,+1.0406] | 74/26 | - | - | nan | nan | 0 |
| grrl_cold | 1.6323 | 0.7965 | 1.5962 | +0.5612 [+0.3709,+0.7760] | 54/46 | -0.2746 [-0.3719,-0.1824] | 32/68 | 46 | 14.9 | 0 |
| grrl_warm | 1.3970 | 0.9282 | 1.3338 | +0.7964 [+0.5983,+1.0020] | 75/25 | -0.0394 [-0.1193,+0.0134] | 1/2 | 99 | 1.3 | 0 |
| ppo_fixed_slot | 1.5187 | 0.9000 | 1.3992 | +0.6748 [+0.5069,+0.8613] | 69/29 | -0.1610 [-0.2446,-0.0839] | 36/64 | 0 | 9.4 | 0 |

Positive Δ = lower loss than the reference (better); negative = degradation.

| method | presence F1 | exact set | exact sequence | pairwise order acc | order given correct set | param MAE |
|---|---|---|---|---|---|---|
| grrl_cold | 0.203 | 0.052 | 0.048 | 0.006 | 0.923 | 0.337 |
| grrl_warm | 0.805 | 0.472 | 0.367 | 0.173 | 0.778 | 0.186 |
| ppo_fixed_slot | 0.352 | 0.048 | 0.032 | 0.080 | 0.667 | 0.220 |
| dry | 0.000 | 0.000 | 0.000 | 0.000 | nan | nan |
| predictor_only | 0.811 | 0.476 | 0.371 | 0.179 | 0.780 | 0.184 |
