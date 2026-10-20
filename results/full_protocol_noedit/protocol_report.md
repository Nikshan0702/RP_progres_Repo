# Held-out evaluation (n=248, dataset/synthetic_v2/test/metadata.jsonl)

| method | mean loss | std | median | vs dry: mean Δ [95% CI] | improved/degraded % | vs predictor: mean Δ [95% CI] | improved/degraded % | stop% | steps | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| dry | 2.1935 | 1.9009 | 1.6164 | - | - | - | - | nan | nan | 0 |
| predictor_only | 1.3577 | 0.7334 | 1.3338 | +0.8358 [+0.6471,+1.0406] | 74/26 | - | - | nan | nan | 0 |
| grrl_warm | 1.3364 | 0.7160 | 1.3268 | +0.8570 [+0.6697,+1.0567] | 75/25 | +0.0212 [+0.0032,+0.0455] | 2/1 | 100 | 1.0 | 0 |

Positive Δ = lower loss than the reference (better); negative = degradation.

| method | presence F1 | exact set | exact sequence | pairwise order acc | order given correct set | param MAE |
|---|---|---|---|---|---|---|
| grrl_warm | 0.805 | 0.468 | 0.367 | 0.179 | 0.784 | 0.184 |
| dry | 0.000 | 0.000 | 0.000 | 0.000 | nan | nan |
| predictor_only | 0.811 | 0.476 | 0.371 | 0.179 | 0.780 | 0.184 |
