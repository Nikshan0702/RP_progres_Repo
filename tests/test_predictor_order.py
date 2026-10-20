"""Learned chain-order prediction: decoder, masking, constraints, metrics, checkpoints."""
import pytest
import torch
import torch.nn.functional as F

from common.checkpoint import IncompatibleCheckpointError
from dataset.dapn_dataset import SUPPORTED_EFFECTS
from models.inverse_fx_net import InverseFXNet, STOP_TOKEN, PAD_TOKEN, encode_chain_tokens
from models.predictor_metrics import edit_distance, parameter_mae, sequence_metrics
from train import order_loss


def test_token_encoding_pads_and_stops():
    t = encode_chain_tokens(["Reverb", "EQ"])
    assert t[:3] == [3, 0, STOP_TOKEN] and set(t[3:]) == {PAD_TOKEN} and len(t) == 8


def test_teacher_forced_logits_mask_used_effects():
    m = InverseFXNet().eval()
    tok = torch.tensor([encode_chain_tokens(["Reverb", "EQ", "Delay"])])
    _, _, lg = m.forward_ordered(torch.randn(1, 1, 44100), torch.randn(1, 1, 44100), tok)
    assert lg[0, 1, 3] < -1e3 and lg[0, 2, 3] < -1e3 and lg[0, 2, 0] < -1e3  # emitted effects unavailable
    assert lg[0, 0].min() > -1e3 and lg[0, 3, 2] < -1e3  # nothing emitted at t=0; Delay used at t=3
    assert lg[0, 3, STOP_TOKEN] > -1e3  # STOP is never masked by the used-set


def test_padding_positions_do_not_contribute_to_loss():
    logits = torch.randn(2, 8, 8)
    tok = torch.tensor([encode_chain_tokens(["EQ"]), encode_chain_tokens(["EQ", "Delay", "Reverb"])])
    base = order_loss(logits, tok)
    logits2 = logits.clone(); logits2[0, 2:] += 100.0  # garbage at PAD positions of sample 0
    assert torch.allclose(base, order_loss(logits2, tok))


def test_decoder_can_learn_non_schema_order_from_audio():
    """Two inputs whose truth is in REVERSE schema order; a fixed-schema/confidence heuristic cannot pass."""
    torch.manual_seed(0)
    m = InverseFXNet(dropout=0.0)
    dry = torch.randn(2, 1, 22050) * 0.1
    wet = torch.stack([dry[0] * 2, dry[1].flip(-1) * 0.3 + 0.05 * torch.randn(22050)])
    seqs = [["Flanger", "Reverb", "EQ"], ["Distortion", "Compressor"]]
    tok = torch.tensor([encode_chain_tokens(s) for s in seqs])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    m.train()
    first = None
    for _ in range(80):
        opt.zero_grad()
        _, _, lg = m.forward_ordered(dry, wet, tok)
        loss = order_loss(lg, tok)
        first = first if first is not None else loss.item()
        loss.backward(); opt.step()
    assert loss.item() < 0.2 * first
    m.eval()
    for i, want in enumerate(seqs):
        got = m.predict_chain(dry[i:i + 1], wet[i:i + 1], min_length=1, max_length=7)["sequence"]
        assert got == want


def test_decoding_constraints_min_max_no_repeat():
    torch.manual_seed(1)
    m = InverseFXNet().eval()
    d, w = torch.randn(1, 1, 22050), torch.randn(1, 1, 22050)
    for lo, hi in [(1, 1), (3, 3), (2, 5), (7, 7)]:
        seq = m.predict_chain(d, w, min_length=lo, max_length=hi)["sequence"]
        assert lo <= len(seq) <= hi and len(set(seq)) == len(seq)


def test_predict_chain_is_valid_for_dapn():
    from dapn.dapn import DAPN
    m = InverseFXNet().eval()
    d, w = torch.randn(1, 1, 22050).clamp(-1, 1) * .1, torch.randn(1, 1, 22050) * .1
    chain = m.predict_chain(d, w, min_length=1, max_length=4)
    assert all(0 <= v <= 1 for s in chain["chain"] for v in s["parameters"].values())
    out = DAPN().process(d, {"chain": chain["chain"]})
    assert torch.isfinite(out).all()


def test_checkpoint_roundtrip_and_legacy_rejected(tmp_path):
    m = InverseFXNet()
    p = tmp_path / "c.pth"
    m.save_checkpoint(p, progress={"epoch": 3}, selection={"best_val_total": 1.0})
    m2 = InverseFXNet(); ck = m2.load_checkpoint(p)
    assert ck["progress"]["epoch"] == 3 and ck["kind"] == "inverse_fx"
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
    with pytest.raises(FileNotFoundError):
        m2.load_checkpoint(tmp_path / "nope.pth")
    with pytest.raises(Exception, match="cannot resume"):
        m2.load_checkpoint(p, mode="resume")  # no optimizer saved -> evaluation-only checkpoint
    legacy = tmp_path / "legacy.pth"
    torch.save({"model_state_dict": m.state_dict(), "epoch": 1}, legacy)
    with pytest.raises(IncompatibleCheckpointError, match="legacy"):
        m2.load_checkpoint(legacy)


def test_sequence_metrics_separate_presence_and_order():
    # right set, wrong order -> presence perfect, order wrong
    r = sequence_metrics([["Reverb", "EQ"]], [["EQ", "Reverb"]])
    assert r["presence_f1"] == 1 and r["presence_set_exact"] == 1
    assert r["order_exact_sequence"] == 0 and r["order_pairwise_acc"] == 0 and r["order_given_correct_set"] == 0
    assert r["order_norm_edit_distance"] == 1.0
    r = sequence_metrics([["EQ", "Reverb"], ["Delay"]], [["EQ", "Reverb"], ["Chorus"]])
    assert r["order_exact_sequence"] == .5 and r["presence_f1"] == .5 and r["order_given_correct_set"] == 1.0
    assert edit_distance(list("abc"), list("axc")) == 1


def test_parameter_mae_only_on_matching_effects():
    pred = [[{"effect": "EQ", "parameters": {"frequency": .5, "gain": .5, "q": .5}}]]
    true = [[{"effect": "EQ", "parameters": {"frequency": .6, "gain": .5, "q": .3}},
             {"effect": "Delay", "parameters": {"delay_time": 0.}}]]
    r = parameter_mae(pred, true)
    assert r["param_count"] == 3 and abs(r["param_mae"] - (.1 + 0 + .2) / 3) < 1e-9
