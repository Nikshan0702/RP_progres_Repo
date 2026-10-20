"""Research protocol plumbing: same pairs for all methods, signed improvements, leakage refusal, splits."""
import csv
import json
from pathlib import Path

import pytest
import torch

from dataset import make_splits
from rl import evaluate_protocol
from rl.grrl_agent import GRRLAgent
from rl.ppo_agent import PPOAgent

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory, v2_predictor_ckpt):
    d = tmp_path_factory.mktemp("p")
    torch.manual_seed(0)
    GRRLAgent().save(d / "cold.pth", extra_config={"warm_start": False})
    GRRLAgent().save(d / "warm.pth", extra_config={"warm_start": True})
    PPOAgent().save(d / "ppo.pth", extra_config={"warm_start": True})
    return d


def subset(tmp_path, split, n):
    rows = [json.loads(l) for l in open(ROOT / f"dataset/synthetic/{split}/metadata.jsonl")][:n]
    p = tmp_path / f"{split}.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows)); return p


def test_protocol_runs_all_methods_on_identical_pairs(tmp_path, ckpts, v2_predictor_ckpt):
    rep = evaluate_protocol.main(["--test-metadata", str(subset(tmp_path, "val", 4)), "--train-metadata", str(subset(tmp_path, "train", 20)),
                                  "--predictor-checkpoint", str(v2_predictor_ckpt), "--grrl-cold-checkpoint", str(ckpts / "cold.pth"),
                                  "--grrl-warm-checkpoint", str(ckpts / "warm.pth"), "--ppo-checkpoint", str(ckpts / "ppo.pth"),
                                  "--num-samples", "3", "--max-chain-length", "3", "--max-episode-steps", "5", "--out-dir", str(tmp_path / "out")])
    assert rep["n_examples"] == 3 and set(rep["methods"]) == {"dry", "predictor_only", "grrl_cold", "grrl_warm", "ppo_fixed_slot"}
    rows = list(csv.DictReader(open(tmp_path / "out" / "per_example.csv")))
    assert len(rows) == 3 and len({r["sample_id"] for r in rows}) == 3
    for r in rows:
        assert all(r[f"{m}_error"] == "" for m in rep["methods"]) and all(float(r[f"{m}_loss"]) >= 0 for m in rep["methods"])
    # predictor-only == warm-start state; RL improvement is predictor - final, signed
    w = rep["methods"]["grrl_warm"]
    for r in rows:
        assert float(r["predictor_only_loss"]) > 0
    imp = w["improvement_vs_predictor"]
    diffs = [float(r["predictor_only_loss"]) - float(r["grrl_warm_loss"]) for r in rows]
    assert imp["mean_improvement"] == pytest.approx(sum(diffs) / 3, abs=1e-6)
    assert imp["pct_improved"] + imp["pct_degraded"] + imp["pct_unchanged"] == pytest.approx(100)
    assert rep["headline"]["rl_improvement_over_predictor_warm"]["n"] == 3 and rep["headline"]["cold_start_improvement_over_dry"]["n"] == 3
    d_imp = rep["methods"]["grrl_cold"]["improvement_vs_dry"]
    dry = [float(r["dry_loss"]) for r in rows]; cold = [float(r["grrl_cold_loss"]) for r in rows]
    assert d_imp["mean_improvement"] == pytest.approx(sum(a - b for a, b in zip(dry, cold)) / 3, abs=1e-6)
    for m in rep["methods"].values():
        assert "presence_f1" in m["chain_metrics"] and "order_pairwise_acc" in m["chain_metrics"]
    assert (tmp_path / "out" / "protocol_report.md").read_text().count("|") > 20
    json.loads((tmp_path / "out" / "protocol_report.json").read_text())


def test_protocol_refuses_train_test_leakage(tmp_path, v2_predictor_ckpt):
    t = subset(tmp_path, "train", 10)
    with pytest.raises(SystemExit, match="overlap"):
        evaluate_protocol.main(["--test-metadata", str(t), "--train-metadata", str(t), "--predictor-checkpoint", str(v2_predictor_ckpt),
                                "--out-dir", str(tmp_path / "o")])


def test_make_splits_is_track_disjoint_deterministic_and_checks_training(tmp_path):
    src = ROOT / "dataset/synthetic/val/metadata.jsonl"
    m1 = make_splits.main(["--metadata", str(src), "--train_metadata", str(ROOT / "dataset/synthetic/train/metadata.jsonl"), "--out", str(tmp_path / "a")])
    m2 = make_splits.main(["--metadata", str(src), "--out", str(tmp_path / "b")])
    assert m1["test_tracks"] == m2["test_tracks"] and not set(m1["test_tracks"]) & set(m1["valsel_tracks"])
    assert m1["test_records"] + m1["valsel_records"] == 500 and m1["test_records"] > 0 and m1["valsel_records"] > 0
    with pytest.raises(SystemExit, match="Leakage"):
        make_splits.main(["--metadata", str(src), "--train_metadata", str(src), "--out", str(tmp_path / "c")])
