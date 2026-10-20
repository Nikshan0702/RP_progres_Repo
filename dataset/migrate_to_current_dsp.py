"""Re-render a dataset's wet audio with the CURRENT (corrected) DAPN. Non-destructive.

The original wet files were rendered by the pre-repair DSP (EQ delayed ~46 ms, reverb with
fixed echo taps, compressor ignoring attack/release, mis-oriented distortion tone FIR, ...).
This script keeps each record's dry audio, effect chain and normalized parameters, renders a new
wet file with today's DAPN into a NEW directory, and writes:
  <out>/<split>/metadata.jsonl   same schema; wet_path -> new file; extra fields
                                 `source_wet_path`, `dsp_fingerprint`, `legacy_vs_new_mrstft`
  <out>/<split>/manifest.json    DSP fingerprint, git commit, counts, mismatch statistics, failures
Originals are never modified/overwritten; the script refuses to write into the source tree.

Usage:
  python -m dataset.migrate_to_current_dsp --metadata dataset/synthetic/val/metadata.jsonl \
      --out dataset/synthetic_v2 --split val [--limit 50]
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from audio_io import load_audio, save_audio
from common.config_loader import get_sample_rate
from common.losses import MultiResolutionSTFTLoss
from dapn.dapn import DAPN


def dsp_fingerprint() -> str:
    h = hashlib.sha256()
    files = sorted((ROOT / "dapn").rglob("*.py")) + [ROOT / "config" / "effects_config.yaml", ROOT / "config" / "system_config.yaml"]
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode()); h.update(f.read_bytes())
    return h.hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", default="dataset/synthetic_v2")
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    out = (ROOT / args.out / args.split).resolve()
    if ROOT / "dataset" / "synthetic" in out.parents or out == ROOT / "dataset" / "synthetic":
        raise SystemExit("Refusing to write inside the original dataset directory")
    if ROOT.resolve() not in out.parents:
        raise SystemExit("--out must be inside the repository (dataset paths are repo-relative)")
    (out / "audio").mkdir(parents=True, exist_ok=True)
    sr = get_sample_rate(); dapn = DAPN(); loss_fn = MultiResolutionSTFTLoss(); fp = dsp_fingerprint()
    records = [json.loads(l) for l in open(args.metadata) if l.strip()][: args.limit]
    mismatches, failures, kept = [], [], 0
    with open(out / "metadata.jsonl", "w") as fh:
        for r in records:
            try:
                dry, _ = load_audio(ROOT / r["dry_path"]); old, _ = load_audio(ROOT / r["wet_path"])
                chain = {"chain": [{"effect": e["effect"], "parameters": e["parameters_normalized"]} for e in r["effects"]]}
                with torch.no_grad():
                    new = dapn.process(dry, chain)
                    if not torch.isfinite(new).all():
                        raise ValueError("nonfinite render")
                    n = min(new.shape[-1], old.shape[-1])
                    mism = float(loss_fn(new[..., :n].reshape(1, 1, -1), old[..., :n].reshape(1, 1, -1)))
                wet_rel = (out / "audio" / f"{r['sample_id']}_wet.wav")
                save_audio(new, wet_rel, sample_rate=sr)
            except Exception as exc:
                failures.append({"sample_id": r["sample_id"], "error": str(exc)}); continue
            rr = dict(r)
            rr.update(source_wet_path=r["wet_path"], wet_path=str(wet_rel.relative_to(ROOT)),
                      dsp_fingerprint=fp, legacy_vs_new_mrstft=mism)
            fh.write(json.dumps(rr) + "\n"); mismatches.append(mism); kept += 1
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = None
    manifest = dict(source_metadata=args.metadata, source_sha256=hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest(),
                    dsp_fingerprint=fp, git_commit=commit, records_in=len(records), records_out=kept, failures=failures,
                    legacy_vs_new_mrstft=dict(mean=statistics.fmean(mismatches) if mismatches else None,
                                              median=statistics.median(mismatches) if mismatches else None,
                                              max=max(mismatches) if mismatches else None),
                    note="dry audio is referenced from the source dataset; only wet audio is re-rendered")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("records_in", "records_out", "legacy_vs_new_mrstft", "dsp_fingerprint")}, indent=1))
    return manifest


if __name__ == "__main__":
    main()
