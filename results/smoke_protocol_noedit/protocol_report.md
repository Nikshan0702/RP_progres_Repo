# Held-out evaluation (n=40, dataset/synthetic_v2_smoke/test/metadata.jsonl)

| method | mean loss | std | median | vs dry: mean Δ [95% CI] | improved/degraded % | vs predictor: mean Δ [95% CI] | improved/degraded % | stop% | steps | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| dry | 2.5892 | 2.2860 | 1.8656 | - | - | - | - | nan | nan | 0 |
| predictor_only | 1.4807 | 0.8819 | 1.3495 | +1.1086 [+0.5168,+1.8303] | 70/30 | - | - | nan | nan | 0 |
| grrl_cold | 1.8608 | 0.7062 | 1.7074 | +0.7284 [+0.2163,+1.3489] | 55/45 | -0.3801 [-0.6754,-0.0740] | 30/70 | 98 | 2.8 | 0 |
| grrl_warm | 1.4494 | 0.8647 | 1.3495 | +1.1398 [+0.5514,+1.8561] | 75/25 | +0.0313 [+0.0000,+0.0828] | 5/0 | 100 | 1.1 | 0 |
| ppo_fixed_slot | 2.3899 | 2.0010 | 1.6674 | +0.1993 [-0.2681,+0.6722] | 42/42 | -0.9093 [-1.5016,-0.4092] | 32/68 | 0 | 9.6 | 0 |

Positive Δ = lower loss than the reference (better); negative = degradation.

| method | presence F1 | exact set | exact sequence | pairwise order acc | order given correct set | param MAE |
|---|---|---|---|---|---|---|
| grrl_cold | 0.270 | 0.125 | 0.125 | 0.000 | 1.000 | 0.257 |
| grrl_warm | 0.706 | 0.425 | 0.400 | 0.125 | 0.941 | 0.213 |
| ppo_fixed_slot | 0.313 | 0.025 | 0.025 | 0.062 | 1.000 | 0.224 |
| dry | 0.000 | 0.000 | 0.000 | 0.000 | nan | nan |
| predictor_only | 0.701 | 0.425 | 0.400 | 0.125 | 0.941 | 0.213 |
