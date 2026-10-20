# Dataset ↔ DSP consistency and migration

## Finding (reproducible)

The wet audio in `dataset/synthetic/{train,val}` was rendered by the **pre-repair DSP**. Re-rendering each
record's stored dry audio + stored normalized chain parameters, and scoring against the stored wet audio with
the multi-resolution STFT loss (lower = closer):

| renderer | mean loss (150 val records) |
|---|---|
| original DSP (`main` @ efb8b74, `git worktree`) | **0.0955** (16-bit quantisation floor) |
| corrected DSP (this branch) | **1.2965** |

Per effect (corrected DSP vs stored wet): Compressor 2.21, EQ 1.90, Reverb 1.66, Distortion 1.59, Delay 1.13,
Flanger 1.06, Chorus 0.92. The ~0.1 floor is the 16-bit storage noise of the log-magnitude term on quiet
passages (the migrated data show the same floor against themselves), so the mismatch is real, not noise.

Defects that changed rendering (see `git diff main -- dapn/`): EQ FIR convolved un-flipped (46.4 ms = 2047-sample
delay at neutral), reverb was a fixed set of echo taps that ignored decay/damping, compressor ignored
attack/release, distortion tone filter was mis-oriented, chorus modulation depth could exceed its base delay.

**Consequence:** any predictor/agent trained on the old wet audio learned a mapping that the corrected DAPN cannot
reproduce (and, for RL, the reward is computed with the corrected DAPN). Old checkpoints are therefore also
incompatible (`common.checkpoint`, schema v2).

## Procedure (non-destructive; originals are never modified)

```bash
# 1. re-render wet audio from stored dry audio + stored chains with the CURRENT DAPN (~0.1 s / sample)
python -m dataset.migrate_to_current_dsp --metadata dataset/synthetic/train/metadata.jsonl --out dataset/synthetic_v2 --split train
python -m dataset.migrate_to_current_dsp --metadata dataset/synthetic/val/metadata.jsonl   --out dataset/synthetic_v2 --split val
# 2. carve a track-disjoint checkpoint-selection split and a final TEST split out of val
python -m dataset.make_splits --metadata dataset/synthetic_v2/val/metadata.jsonl \
    --train_metadata dataset/synthetic_v2/train/metadata.jsonl --out dataset/synthetic_v2
```

Outputs per split: `metadata.jsonl` (same schema + `source_wet_path`, `dsp_fingerprint`, `legacy_vs_new_mrstft`),
`manifest.json` (DSP fingerprint, git commit, source hash, mismatch statistics, failures), `audio/*_wet.wav`.
Dry audio is referenced from the source dataset. The migration refuses to write inside `dataset/synthetic/` or outside the repo.
`dsp_fingerprint` (hash of `dapn/**/*.py` + both YAML configs) lets training/evaluation detect data rendered by a different DSP.

Cost (measured here): ≈ 0.1 s/record → ≈ 8 min and ≈ 0.8 GB for the 4 500 records.

## Status

The full migration has **not** been run automatically. Smoke-scale subsets (600 train / 120 val) were migrated into
`dataset/synthetic_v2_smoke/` to verify the pipeline end to end; they are not sufficient for research claims.

## Remaining dataset limitations

* Source tracks: 85 train / 15 val, disjoint. There is no separate test split in the shipped data; `dataset/make_splits.py` creates one by source track.
* Chains have 1–3 effects (generator default), so order labels for ≥4-effect chains are unseen by the predictor.
* Stored `parameters_normalized` are rounded to 4 decimals; wet audio is 16-bit. Re-rendering is therefore reproducible only up to that precision.
