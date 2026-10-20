# Held-out evaluation (n=40, dataset/synthetic_v2_smoke/test/metadata.jsonl)

| method | mean loss | std | median | vs dry: mean Δ [95% CI] | improved/degraded % | vs predictor: mean Δ [95% CI] | improved/degraded % | stop% | steps | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| dry | 2.5892 | 2.2860 | 1.8656 | - | - | - | - | nan | nan | 0 |
| predictor_only | 1.4807 | 0.8819 | 1.3495 | +1.1086 [+0.5168,+1.8303] | 70/30 | - | - | nan | nan | 0 |
| grrl_cold | 1.7274 | 0.8439 | 1.4877 | +0.8619 [+0.3193,+1.5301] | 58/42 | -0.2467 [-0.4713,-0.0341] | 38/62 | 100 | 2.0 | 0 |
| grrl_warm | 1.4778 | 0.8790 | 1.3495 | +1.1115 [+0.5241,+1.8336] | 72/28 | +0.0029 [-0.0192,+0.0280] | 2/2 | 100 | 1.5 | 0 |

Positive Δ = lower loss than the reference (better); negative = degradation.

| method | presence F1 | exact set | exact sequence | pairwise order acc | order given correct set | param MAE |
|---|---|---|---|---|---|---|
| grrl_cold | 0.227 | 0.050 | 0.050 | 0.000 | 1.000 | 0.220 |
| grrl_warm | 0.697 | 0.425 | 0.400 | 0.125 | 0.941 | 0.213 |
| dry | 0.000 | 0.000 | 0.000 | 0.000 | nan | nan |
| predictor_only | 0.701 | 0.425 | 0.400 | 0.125 | 0.941 | 0.213 |
