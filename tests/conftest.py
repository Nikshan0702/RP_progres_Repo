"""tests/conftest.py — configure path for all tests."""
import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import pytest


@pytest.fixture(scope="session")
def v2_predictor_ckpt(tmp_path_factory):
    """Randomly initialised (untrained) schema-v2 predictor checkpoint, for plumbing tests only."""
    import torch
    from models.inverse_fx_net import InverseFXNet
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "predictor_v2.pth"
    InverseFXNet().save_checkpoint(path, progress={"epoch": 0}, selection={"note": "untrained test fixture"})
    return path
