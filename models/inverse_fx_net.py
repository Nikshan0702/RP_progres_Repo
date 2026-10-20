"""
models/inverse_fx_net.py
========================
Deep Convolutional Neural Network for Inverse Audio Effects Chain Prediction.

Architecture:
  Input: (dry_audio, wet_audio), each shape (B, 1, T) at 44.1 kHz.
  1. Front-end: Computes Log-Mel Spectrograms for dry, wet, and delta (wet - dry).
     Feature tensor: (B, 3, n_mels, time_frames).
  2. Backbone: 4-stage 2D Residual Convolutional Network extracting acoustic representations.
  3. Pooling: Concatenated Global Average & Max Pooling (GAP + GMP) -> Dense Embedding.
  4. Multi-Task Heads:
     - presence_head: (B, 7) Logits for active effect classification (BCE loss).
     - param_head: (B, 29) Continuous normalized parameter predictions in [0, 1] (Sigmoid).
     - order decoder (v2): GRU sequence decoder conditioned on the embedding. At step t it
       predicts the t-th effect of the chain or STOP (token 7), trained with teacher forcing
       and cross-entropy; already-emitted effects are masked (no repeats). BOS (token 8) is
       an input-only token. The decoded sequence IS the predicted chain order; there is no
       confidence sorting or fixed schema order.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchaudio.transforms as T

from common.checkpoint import load_checkpoint, save_checkpoint

from dataset.dapn_dataset import SUPPORTED_EFFECTS, PARAM_SCHEMA, PARAM_OFFSETS, TOTAL_NUM_PARAMS


NUM_EFFECTS = len(SUPPORTED_EFFECTS)
STOP_TOKEN = NUM_EFFECTS          # 7: end of chain (decoder output)
BOS_TOKEN = NUM_EFFECTS + 1       # 8: decoder start (input only)
PAD_TOKEN = -100                  # ignored by the order loss
MAX_SEQ_LEN = NUM_EFFECTS + 1     # all effects once + STOP
ORDER_DECODER_NAME = "gru_v1"


def encode_chain_tokens(chain_sequence: list[str]) -> list[int]:
    """Effect names -> fixed-length target tokens: effects, STOP, then PAD."""
    toks = [SUPPORTED_EFFECTS.index(n) for n in chain_sequence] + [STOP_TOKEN]
    return toks + [PAD_TOKEN] * (MAX_SEQ_LEN - len(toks))


class ConvBlock(nn.Module):
    """Residual 2D Convolutional Block with BatchNorm and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # Residual shortcut if dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out


class InverseFXNet(nn.Module):
    """End-to-end Inverse FX Predictor predicting active effects and continuous DSP parameters."""

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate

        # 1. Front-end Mel-spectrogram extractor
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        # 2. 2D Convolutional Backbone (3 input channels: dry, wet, delta)
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, stride=2),    # (B, 32, 32, 87)
            ConvBlock(32, 64, stride=2),   # (B, 64, 16, 44)
            ConvBlock(64, 128, stride=2),  # (B, 128, 8, 22)
            ConvBlock(128, 256, stride=2), # (B, 256, 4, 11)
        )

        # 3. Dense Bottleneck (GAP + GMP = 256 * 2 = 512)
        self.fc = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )

        self.embedding_dim = embedding_dim
        self.n_fft, self.hop_length, self.n_mels, self.dropout_p = n_fft, hop_length, n_mels, dropout

        # 4. Order decoder (sequence prediction with STOP)
        self.token_embed = nn.Embedding(NUM_EFFECTS + 2, 64)
        self.decoder_init = nn.Linear(embedding_dim, embedding_dim)
        self.decoder = nn.GRU(64 + embedding_dim, embedding_dim, batch_first=True)
        self.order_head = nn.Linear(embedding_dim, NUM_EFFECTS + 1)

        # 5. Multi-Task Output Heads
        # Head A: Effect presence logits (7 effects)
        self.presence_head = nn.Linear(embedding_dim, len(SUPPORTED_EFFECTS))

        # Head B: Continuous parameter values (29 parameters in [0, 1])
        self.param_head = nn.Sequential(
            nn.Linear(embedding_dim, TOTAL_NUM_PARAMS),
            nn.Sigmoid(),
        )

    def extract_features(self, dry_audio: torch.Tensor, wet_audio: torch.Tensor) -> torch.Tensor:
        """Extracts 3-channel Log-Mel spectrograms: [Dry, Wet, Delta]."""
        eps = 1e-6
        # dry_audio: (B, 1, T), wet_audio: (B, 1, T)
        mel_dry = torch.log(self.mel_transform(dry_audio) + eps)
        mel_wet = torch.log(self.mel_transform(wet_audio) + eps)
        mel_delta = mel_wet - mel_dry

        # Concatenate along channel dimension: (B, 3, n_mels, frames)
        features = torch.cat([mel_dry, mel_wet, mel_delta], dim=1)
        return features

    def encode(self, dry_audio: torch.Tensor, wet_audio: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(dry_audio, wet_audio)
        feats = self.backbone(x)  # (B, 256, H, W)
        pooled = torch.cat([feats.mean(dim=[-2, -1]), feats.amax(dim=[-2, -1])], dim=-1)  # GAP+GMP
        return self.fc(pooled)

    def forward(
        self,
        dry_audio: torch.Tensor,
        wet_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Presence logits (B, 7) and normalized parameters (B, 29) in [0, 1]."""
        emb = self.encode(dry_audio, wet_audio)
        return self.presence_head(emb), self.param_head(emb)

    def forward_ordered(self, dry_audio, wet_audio, target_tokens: torch.Tensor):
        """Teacher-forced pass. ``target_tokens``: (B, MAX_SEQ_LEN) effects, STOP, PAD(-100).

        Returns ``(presence_logits, params, order_logits)`` with order_logits (B, MAX_SEQ_LEN, 8);
        used effects are masked to -inf-like so the loss is consistent with constrained decoding.
        """
        emb = self.encode(dry_audio, wet_audio)
        B = emb.shape[0]
        prev = torch.cat([torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=emb.device),
                          target_tokens[:, :-1].clamp(min=0)], dim=1)  # PAD -> any valid id (masked in loss)
        h0 = torch.tanh(self.decoder_init(emb)).unsqueeze(0)
        inp = torch.cat([self.token_embed(prev), emb.unsqueeze(1).expand(-1, prev.shape[1], -1)], dim=-1)
        out, _ = self.decoder(inp, h0.contiguous())
        logits = self.order_head(out)
        onehot = torch.zeros(B, prev.shape[1], NUM_EFFECTS + 1, device=emb.device)
        eff_prev = (target_tokens[:, :-1] >= 0) & (target_tokens[:, :-1] < NUM_EFFECTS)
        scatter_idx = target_tokens[:, :-1].clamp(0, NUM_EFFECTS)
        onehot[:, 1:].scatter_(2, scatter_idx.unsqueeze(-1), eff_prev.unsqueeze(-1).float())
        used = onehot.cumsum(dim=1) > 0  # effects emitted strictly before position t
        used[..., STOP_TOKEN] = False
        return self.presence_head(emb), self.param_head(emb), logits.masked_fill(used, -1e4)

    @torch.no_grad()
    def decode_order(self, emb: torch.Tensor, min_length: int = 1, max_length: int = NUM_EFFECTS):
        """Greedy constrained decoding. Returns (tokens list[list[int]], step probabilities).

        STOP is masked until ``min_length`` effects were emitted and forced at ``max_length``;
        emitted effects are masked (no repeats). Everything else is the learned distribution.
        """
        B = emb.shape[0]
        h = torch.tanh(self.decoder_init(emb)).unsqueeze(0).contiguous()
        prev = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=emb.device)
        used = torch.zeros(B, NUM_EFFECTS + 1, dtype=torch.bool, device=emb.device)
        done = torch.zeros(B, dtype=torch.bool, device=emb.device)
        seqs: list[list[int]] = [[] for _ in range(B)]
        probs: list[list[float]] = [[] for _ in range(B)]
        for t in range(min(max_length, NUM_EFFECTS) + 1):
            out, h = self.decoder(torch.cat([self.token_embed(prev), emb.unsqueeze(1)], dim=-1), h)
            logits = self.order_head(out[:, 0]).masked_fill(used, float("-inf"))
            if t < min_length:
                logits[:, STOP_TOKEN] = float("-inf")
            if t >= max_length:
                logits[:, :NUM_EFFECTS] = float("-inf")
            p = torch.softmax(logits, dim=-1)
            tok = p.argmax(dim=-1)
            for i in range(B):
                if not done[i]:
                    if int(tok[i]) == STOP_TOKEN:
                        done[i] = True
                    else:
                        seqs[i].append(int(tok[i]))
                        probs[i].append(float(p[i, tok[i]]))
            used = used | (torch.nn.functional.one_hot(tok, NUM_EFFECTS + 1).bool() & (tok < NUM_EFFECTS).unsqueeze(-1))
            prev = tok.unsqueeze(1)
            if bool(done.all()):
                break
        return seqs, probs

    def predict_chain(
        self,
        dry_audio: torch.Tensor,
        wet_audio: torch.Tensor,
        min_length: int = 1,
        max_length: int = NUM_EFFECTS,
    ) -> dict:
        """Decode an ordered chain {effect, parameters}. Parameters are the model outputs
        for the decoded effects (unrounded, so the predictor-only baseline is reproducible).
        Never fabricates an identity-EQ fallback: a valid chain needs >= min_length effects,
        which constrained decoding guarantees."""
        self.eval()
        with torch.no_grad():
            if dry_audio.ndim == 2:
                dry_audio = dry_audio.unsqueeze(0)
            if wet_audio.ndim == 2:
                wet_audio = wet_audio.unsqueeze(0)
            emb = self.encode(dry_audio, wet_audio)
            presence = torch.sigmoid(self.presence_head(emb))[0]
            params = self.param_head(emb)[0]
            seqs, probs = self.decode_order(emb[:1], min_length=min_length, max_length=max_length)
        chain = []
        for tok in seqs[0]:
            name = SUPPORTED_EFFECTS[tok]
            lo, _ = PARAM_OFFSETS[name]
            chain.append({"effect": name,
                          "parameters": {p: float(params[lo + i]) for i, p in enumerate(PARAM_SCHEMA[name])}})
        return {"chain": chain, "sequence": [SUPPORTED_EFFECTS[t] for t in seqs[0]],
                "order_confidence": probs[0], "presence_probabilities": presence.tolist()}

    # ------------------------------------------------------------------
    # Checkpointing (schema v2; legacy best_model.pth has no order decoder)
    # ------------------------------------------------------------------
    def config(self) -> dict:
        return {"sample_rate": self.sample_rate, "n_fft": self.n_fft, "hop_length": self.hop_length,
                "n_mels": self.n_mels, "embedding_dim": self.embedding_dim, "dropout": self.dropout_p,
                "order_decoder": ORDER_DECODER_NAME, "effects": list(SUPPORTED_EFFECTS),
                "max_seq_len": MAX_SEQ_LEN}

    def save_checkpoint(self, path, *, optimizer=None, scheduler=None, progress=None, selection=None,
                        provenance=None) -> None:
        save_checkpoint(path, kind="inverse_fx", config=self.config(), model_state=self.state_dict(),
                        optimizer_state=optimizer.state_dict() if optimizer else None,
                        scheduler_state=scheduler.state_dict() if scheduler else None,
                        progress=progress, selection=selection, provenance=provenance)

    def load_checkpoint(self, path, mode: str = "eval") -> dict:
        """Strictly load a v2 checkpoint; legacy/incompatible files raise IncompatibleCheckpointError."""
        cfg = self.config()
        ckpt = load_checkpoint(path, kind="inverse_fx", mode=mode,
                               expected_config={k: cfg[k] for k in ("embedding_dim", "n_mels", "order_decoder", "effects")})
        self.load_state_dict(ckpt["model_state"])
        return ckpt
