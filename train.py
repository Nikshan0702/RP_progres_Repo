"""
train.py
========
Training pipeline for the Inverse FX Chain Predictor (InverseFXNet v2).

Objectives (reported SEPARATELY, never mixed into one "accuracy"):
  1. presence : multi-label BCE over the 7 effects.
  2. params   : masked MSE over parameters of effects that are present.
  3. order    : teacher-forced sequence cross-entropy over (effects..., STOP) with used-effect
                masking; PAD positions are ignored (variable-length chains).
Total = presence + 2*params + order_weight*order.

Validation computes held-out presence/order/parameter metrics from DECODED chains
(``predict_chain`` logic, greedy constrained decoding), not from teacher-forced logits.
The best checkpoint is selected on validation order+presence+param composite loss
(``selection_score``); training data never enters validation.

Usage:
  python3 train.py --epochs 20 --batch_size 16
  python3 train.py --resume checkpoints_v2/last_model.pth --epochs 30
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from common.checkpoint import dataset_provenance, restore_rng_state
from dataset.dapn_dataset import PARAM_OFFSETS, PARAM_SCHEMA, SUPPORTED_EFFECTS, get_dataloader
from models.inverse_fx_net import InverseFXNet, STOP_TOKEN
from models.predictor_metrics import parameter_mae, sequence_metrics


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_masked_param_loss(preds, targets, mask):
    """MSE only on parameters corresponding to active effects."""
    return ((preds - targets).pow(2) * mask).sum() / (mask.sum() + 1e-8)


def compute_param_mae(preds, targets, mask) -> float:
    return float((((preds - targets).abs() * mask).sum() / (mask.sum() + 1e-8)).item())


def order_loss(order_logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over (effects..., STOP); PAD (-100) ignored."""
    return F.cross_entropy(order_logits.reshape(-1, order_logits.shape[-1]), tokens.reshape(-1), ignore_index=-100)


def batch_losses(model, batch, device, order_weight):
    dry, wet = batch["dry_audio"].to(device), batch["wet_audio"].to(device)
    pres_t, par_t = batch["presence_multihot"].to(device), batch["param_targets"].to(device)
    par_m, tokens = batch["param_mask"].to(device), batch["order_tokens"].to(device)
    logits, params, order_logits = model.forward_ordered(dry, wet, tokens)
    lp = F.binary_cross_entropy_with_logits(logits, pres_t)
    lm = compute_masked_param_loss(params, par_t, par_m)
    lo = order_loss(order_logits, tokens)
    return lp, lm, lo, lp + 2.0 * lm + order_weight * lo, (dry, wet)


@torch.no_grad()
def validate(model, loader, device, order_weight):
    """Held-out loss components + decoded-chain presence/order/parameter metrics."""
    model.eval()
    sums = dict(presence=0., param=0., order=0., total=0.)
    pred_seqs, true_seqs, pred_chains, true_effects = [], [], [], []
    for batch in loader:
        lp, lm, lo, lt, (dry, wet) = batch_losses(model, batch, device, order_weight)
        sums["presence"] += lp.item(); sums["param"] += lm.item(); sums["order"] += lo.item(); sums["total"] += lt.item()
        emb = model.encode(dry, wet)
        params = model.param_head(emb)
        seqs, _ = model.decode_order(emb, min_length=1, max_length=len(SUPPORTED_EFFECTS))
        for i, seq in enumerate(seqs):
            names = [SUPPORTED_EFFECTS[t] for t in seq]
            pred_seqs.append(names)
            pred_chains.append([{"effect": n, "parameters": {p: float(params[i, PARAM_OFFSETS[n][0] + k])
                                                             for k, p in enumerate(PARAM_SCHEMA[n])}} for n in names])
        true_seqs.extend(batch["chain_sequence"])
        # parameters_normalized from the dataset targets
        for i, seq in enumerate(batch["chain_sequence"]):
            true_effects.append([{"effect": n, "parameters": {p: float(batch["param_targets"][i, PARAM_OFFSETS[n][0] + k])
                                                             for k, p in enumerate(PARAM_SCHEMA[n])}} for n in seq])
    nb = len(loader)
    out = {f"val_loss_{k}": v / nb for k, v in sums.items()}
    out.update({f"val_{k}": v for k, v in sequence_metrics(pred_seqs, true_seqs).items()})
    out.update({f"val_{k}": v for k, v in parameter_mae(pred_chains, true_effects).items()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train InverseFXNet v2 (presence + parameters + learned order)")
    ap.add_argument("--train_metadata", default="dataset/synthetic/train/metadata.jsonl")
    ap.add_argument("--val_metadata", default="dataset/synthetic/val/metadata.jsonl")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--order_weight", type=float, default=1.0)
    ap.add_argument("--save_dir", default="checkpoints_v2", help="Never defaults to the legacy 'checkpoints/' directory")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None, help="Resume training from a v2 checkpoint (must exist)")
    ap.add_argument("--max_train_batches", type=int, default=None, help="Smoke-test limit")
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    if save_dir.resolve() == (ROOT / "checkpoints").resolve():
        raise SystemExit("Refusing to write into legacy checkpoints/; choose another --save_dir")
    save_dir.mkdir(parents=True, exist_ok=True)
    legacy_unsafe = ROOT / "checkpoints" / "best_model.pth"

    train_loader = get_dataloader(args.train_metadata, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(args.val_metadata, batch_size=args.batch_size, shuffle=False)
    for key in ("dry_path", "wet_path", "source_track"):  # sample_ids restart per split; paths/tracks are the real identity
        if {r[key] for r in train_loader.dataset.records} & {r[key] for r in val_loader.dataset.records}:
            raise SystemExit(f"Train/val overlap on '{key}': refusing to train (data leakage)")
    print(f"device={device} train={len(train_loader.dataset)} val={len(val_loader.dataset)} seed={args.seed}")

    model = InverseFXNet().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    start_epoch, best, history = 1, float("inf"), []
    if args.resume:
        ckpt = model.load_checkpoint(args.resume, mode="resume")  # FileNotFoundError / Incompatible... if invalid
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        restore_rng_state(ckpt["rng_state"])
        start_epoch = ckpt["progress"]["epoch"] + 1
        best = ckpt["selection"].get("best_val_total", float("inf"))
        print(f"Resumed at epoch {start_epoch} (best val total so far {best:.4f})")

    provenance = dict(datasets=dataset_provenance(args.train_metadata, args.val_metadata), seed=args.seed,
                      args=vars(args), note="Trained on dataset wet audio as stored; see docs/DATASET_MIGRATION.md")
    log_path = save_dir / "predictor_training_log.jsonl"

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter(); model.train()
        acc = dict(presence=0., param=0., order=0., total=0.); n = 0
        for bi, batch in enumerate(train_loader):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            optimizer.zero_grad()
            lp, lm, lo, loss, _ = batch_losses(model, batch, device, args.order_weight)
            loss.backward()
            optimizer.step()
            for k, v in zip(acc, (lp, lm, lo, loss)):
                acc[k] += v.item()
            n += 1
        scheduler.step()
        val = validate(model, val_loader, device, args.order_weight)
        entry = {"epoch": epoch, "seconds": round(time.perf_counter() - t0, 1),
                 **{f"train_loss_{k}": v / max(n, 1) for k, v in acc.items()}, **val}
        history.append(entry)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        is_best = val["val_loss_total"] < best
        if is_best:
            best = val["val_loss_total"]
        sel = dict(best_val_total=best, metric="val_loss_total (presence+2*param+order_weight*order)",
                   epoch_of_best=epoch if is_best else None)
        prog = dict(epoch=epoch, train_samples_seen=n * args.batch_size)
        model.save_checkpoint(save_dir / "last_model.pth", optimizer=optimizer, scheduler=scheduler,
                              progress=prog, selection=sel, provenance=provenance)
        if is_best:
            model.save_checkpoint(save_dir / "best_model.pth", optimizer=optimizer, scheduler=scheduler,
                                  progress=prog, selection=sel, provenance=provenance)
        print(f"[{epoch:02d}/{args.epochs}] {entry['seconds']}s train(total={entry['train_loss_total']:.3f} "
              f"pres={entry['train_loss_presence']:.3f} par={entry['train_loss_param']:.3f} ord={entry['train_loss_order']:.3f}) | "
              f"val(F1={val['val_presence_f1']:.3f} paramMAE={val['val_param_mae']:.3f} "
              f"seqExact={val['val_order_exact_sequence']:.3f} pairAcc={val['val_order_pairwise_acc']:.3f} "
              f"ordGivenSet={val['val_order_given_correct_set']:.3f}){' *best' if is_best else ''}")
    print(f"Best checkpoint: {(save_dir / 'best_model.pth').resolve()}")
    return history


if __name__ == "__main__":
    main()
