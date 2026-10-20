"""Carve a track-disjoint checkpoint-selection split and a final TEST split from one metadata file.

The synthetic dataset ships only train/val. Using val for BOTH checkpoint selection and final claims
would tune on the test set, so this script deterministically assigns each *source track* to either
``valsel`` (used for early stopping / best-checkpoint selection) or ``test`` (touched only by
``rl.evaluate_protocol``). It also verifies that no source track appears in the training metadata.

  python -m dataset.make_splits --metadata dataset/synthetic_v2/val/metadata.jsonl \
      --train_metadata dataset/synthetic_v2/train/metadata.jsonl --out dataset/synthetic_v2 --test_fraction 0.5
"""
import argparse
import hashlib
import json
from pathlib import Path


def assign(track: str, seed: int) -> float:
    return int(hashlib.sha256(f"{seed}:{track}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--train_metadata", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test_fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(args.metadata) if l.strip()]
    tracks = sorted({r["source_track"] for r in rows})
    # rank by hash so the split is deterministic and independent of record order
    ranked = sorted(tracks, key=lambda t: assign(t, args.seed))
    n_test = max(1, round(args.test_fraction * len(tracks)))
    test_tracks, sel_tracks = set(ranked[:n_test]), set(ranked[n_test:])
    if not sel_tracks:
        raise SystemExit("test_fraction leaves no tracks for checkpoint selection")
    if args.train_metadata:
        train_tracks = {json.loads(l)["source_track"] for l in open(args.train_metadata) if l.strip()}
        if train_tracks & set(tracks):
            raise SystemExit(f"Leakage: {len(train_tracks & set(tracks))} source tracks also appear in training data")
    out = Path(args.out)
    manifest = {"source": args.metadata, "source_sha256": hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest(), "seed": args.seed,
                "test_fraction": args.test_fraction, "test_tracks": sorted(test_tracks), "valsel_tracks": sorted(sel_tracks)}
    for name, keep in (("test", test_tracks), ("valsel", sel_tracks)):
        d = out / name; d.mkdir(parents=True, exist_ok=True)
        sub = [r for r in rows if r["source_track"] in keep]
        (d / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in sub) + "\n")
        manifest[f"{name}_records"] = len(sub)
    (out / "splits_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("tracks")}, indent=1))
    return manifest


if __name__ == "__main__":
    main()
