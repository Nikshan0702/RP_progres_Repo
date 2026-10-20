"""Dataset/DSP consistency and the non-destructive migration procedure."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch

from audio_io import load_audio
from common.losses import MultiResolutionSTFTLoss
from dapn.dapn import DAPN
from dataset import migrate_to_current_dsp as mig

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dataset/synthetic/val/metadata.jsonl"


@pytest.fixture
def out_dir():
    d = ROOT / "dataset" / "_pytest_migration"
    shutil.rmtree(d, ignore_errors=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_migration_is_nondestructive_and_documents_the_mismatch(out_dir):
    before = hashlib.sha256(SRC.read_bytes()).hexdigest()
    first_wet = [json.loads(l) for l in open(SRC)][0]["wet_path"]
    wet_hash = hashlib.sha256((ROOT / first_wet).read_bytes()).hexdigest()
    man = mig.main(["--metadata", str(SRC), "--out", str(out_dir.relative_to(ROOT)), "--split", "val", "--limit", "6"])
    assert man["records_out"] == 6 and not man["failures"] and man["dsp_fingerprint"]
    # the stored wet audio really differs from the corrected DSP (evidence that old data are inconsistent)
    assert man["legacy_vs_new_mrstft"]["mean"] > 0.3
    # originals untouched
    assert hashlib.sha256(SRC.read_bytes()).hexdigest() == before
    assert hashlib.sha256((ROOT / first_wet).read_bytes()).hexdigest() == wet_hash
    rows = [json.loads(l) for l in open(out_dir / "val" / "metadata.jsonl")]
    assert rows[0]["wet_path"].startswith("dataset/_pytest_migration/") and rows[0]["source_wet_path"] == first_wet
    assert all(r["dsp_fingerprint"] == man["dsp_fingerprint"] and r["chain_sequence"] for r in rows)
    # re-rendering with the current DAPN reproduces the migrated wet up to 16-bit quantisation. The MR-STFT log
    # term makes that noise floor ~0.08-0.1 on quiet passages (same floor as old-DSP vs old-wet), vs ~1.2 mismatch.
    d, w = DAPN(), MultiResolutionSTFTLoss()
    r = rows[0]
    dry, _ = load_audio(ROOT / r["dry_path"]); wet, _ = load_audio(ROOT / r["wet_path"])
    chain = {"chain": [{"effect": e["effect"], "parameters": e["parameters_normalized"]} for e in r["effects"]]}
    with torch.no_grad():
        again = d.process(dry, chain)
        assert float(w(again.reshape(1, 1, -1), wet.reshape(1, 1, -1))) < 0.15
        assert (again - wet).abs().max() < 1e-3


def test_migration_refuses_to_overwrite_original_dataset():
    with pytest.raises(SystemExit):
        mig.main(["--metadata", str(SRC), "--out", "dataset/synthetic", "--split", "val", "--limit", "1"])
    with pytest.raises(SystemExit):
        mig.main(["--metadata", str(SRC), "--out", "/tmp/elsewhere", "--split", "val", "--limit", "1"])


def test_fingerprint_changes_when_dsp_changes(monkeypatch):
    a = mig.dsp_fingerprint()
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: real(self) + (b"x" if self.name == "eq.py" else b""))
    assert mig.dsp_fingerprint() != a
