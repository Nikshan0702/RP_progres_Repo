"""
evaluate_summary.py
===================
Held-out evaluation of an ordered InverseFXNet checkpoint, with presence, order and parameter
metrics reported SEPARATELY, plus closed-loop reconstruction (predicted chain rendered by the
corrected DAPN vs. the target and vs. the dry baseline; improvements are signed).

Usage:
  python3 evaluate_summary.py --checkpoint checkpoints_v2/best_model.pth --val_metadata <metadata.jsonl> --num_recon 50
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from common.losses import MultiResolutionSTFTLoss
from dapn.dapn import DAPN
from dataset.dapn_dataset import get_dataloader
from models.inverse_fx_net import InverseFXNet
from train import validate


def main(argv=None):
    ap = argparse.ArgumentParser(description="Held-out evaluation of the ordered predictor")
    ap.add_argument("--checkpoint", default="checkpoints_v2/best_model.pth")
    ap.add_argument("--val_metadata", default="dataset/synthetic/val/metadata.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_recon", type=int, default=50)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    model = InverseFXNet().to(device)
    ck = model.load_checkpoint(args.checkpoint)   # raises IncompatibleCheckpointError for legacy files
    loader = get_dataloader(args.val_metadata, batch_size=args.batch_size, shuffle=False)
    m = validate(model, loader, device, order_weight=1.0)

    dapn, loss_fn = DAPN(), MultiResolutionSTFTLoss()
    ds, rows = loader.dataset, []
    for i in range(min(args.num_recon, len(ds))):
        s = ds[i]
        dry, wet = s["dry_audio"].unsqueeze(0).to(device), s["wet_audio"].unsqueeze(0).to(device)
        chain = model.predict_chain(dry, wet)
        with torch.no_grad():
            out = dapn.process(dry, {"chain": chain["chain"]})
            rows.append({"sample_id": s["sample_id"], "dry_loss": float(loss_fn(dry, wet)), "pred_loss": float(loss_fn(out, wet)),
                         "pred_sequence": chain["sequence"], "true_sequence": s["chain_sequence"]})
    imp = [r["dry_loss"] - r["pred_loss"] for r in rows]
    report = {"checkpoint": args.checkpoint, "checkpoint_selection": ck["selection"], "n_val": len(ds), "metrics": m,
              "closed_loop": {"n": len(rows), "mean_dry_loss": statistics.fmean(r["dry_loss"] for r in rows),
                              "mean_pred_loss": statistics.fmean(r["pred_loss"] for r in rows),
                              "mean_improvement_vs_dry": statistics.fmean(imp), "pct_improved": 100 * sum(d > 0 for d in imp) / len(imp),
                              "pct_degraded": 100 * sum(d < 0 for d in imp) / len(imp)}, "per_example": rows}
    print(json.dumps({k: v for k, v in report.items() if k != "per_example"}, indent=2, default=float))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
