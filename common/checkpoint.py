"""Versioned, validated checkpoints shared by the predictor, GRRL and fixed-slot PPO.

A checkpoint is a dict with ``schema_version``, ``kind`` (e.g. ``grrl``), the model
``config`` (everything needed to rebuild/validate the architecture), ``model_state``,
optional ``optimizer_state``/``scheduler_state``, ``progress`` (timesteps, updates,
epoch), ``rng_state`` (python/numpy/torch), ``selection`` (best-validation metadata)
and ``provenance`` (dataset manifest hash, library versions, git commit).

Loading distinguishes two modes:
  * ``mode="eval"``   - weights only; optimizer/RNG are ignored.
  * ``mode="resume"`` - additionally requires optimizer + progress and returns them.
Legacy (pre-schema) files are never silently accepted: they raise
:class:`IncompatibleCheckpointError` unless ``allow_legacy=True`` is passed, which
only returns the raw dict for explicit, caller-side migration.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA_VERSION = 2


class CheckpointError(RuntimeError):
    pass


class IncompatibleCheckpointError(CheckpointError):
    pass


def capture_rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def file_sha256(path: str | Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(limit) if limit else fh.read())
    return h.hexdigest()


def dataset_provenance(*metadata_paths: str | Path | None) -> dict[str, Any]:
    out = {}
    for p in metadata_paths:
        if p and Path(p).exists():
            out[str(p)] = {"sha256": file_sha256(p), "records": sum(1 for _ in open(p, "rb"))}
    return out


def environment_info() -> dict[str, Any]:
    info = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform()}
    try:
        import torchaudio
        info["torchaudio"] = torchaudio.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        root = Path(__file__).resolve().parent.parent
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                                     stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        info["git_commit"] = None
    return info


def save_checkpoint(path: str | Path, *, kind: str, config: dict, model_state: dict,
                    optimizer_state: dict | None = None, scheduler_state: dict | None = None,
                    progress: dict | None = None, selection: dict | None = None,
                    provenance: dict | None = None, rng_state: dict | None = None,
                    overwrite: bool = True) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise CheckpointError(f"Refusing to overwrite existing checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION, "kind": kind, "config": config,
        "model_state": model_state, "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state, "progress": progress or {},
        "rng_state": rng_state if rng_state is not None else capture_rng_state(),
        "selection": selection or {}, "provenance": {**(provenance or {}), "environment": environment_info()},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a crash never leaves a truncated checkpoint under the final name
    return path


def load_checkpoint(path: str | Path, *, kind: str, expected_config: dict | None = None,
                    mode: str = "eval", map_location: str | torch.device = "cpu",
                    allow_legacy: bool = False) -> dict[str, Any]:
    if mode not in ("eval", "resume"):
        raise ValueError("mode must be 'eval' or 'resume'")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict):
        raise IncompatibleCheckpointError(f"{path}: not a checkpoint dictionary")
    if ckpt.get("schema_version") != SCHEMA_VERSION:
        if allow_legacy:
            return ckpt
        raise IncompatibleCheckpointError(
            f"{path}: legacy/unversioned checkpoint (schema_version={ckpt.get('schema_version')!r}, "
            f"need {SCHEMA_VERSION}). It predates the corrected DSP/policy/predictor and carries no "
            f"order-prediction or bounded-policy weights; retrain or migrate explicitly.")
    if ckpt["kind"] != kind:
        raise IncompatibleCheckpointError(f"{path}: kind '{ckpt['kind']}' != expected '{kind}'")
    for key, want in (expected_config or {}).items():
        have = ckpt["config"].get(key)
        if have != want:
            raise IncompatibleCheckpointError(f"{path}: config[{key!r}]={have!r} but model expects {want!r}")
    if mode == "resume":
        missing = [k for k in ("optimizer_state", "progress") if not ckpt.get(k)]
        if missing:
            raise CheckpointError(f"{path}: cannot resume, missing {missing} (evaluation-only checkpoint?)")
    return ckpt


def describe(path: str | Path) -> dict[str, Any]:
    """Compatibility report that never raises on legacy files (for CLIs/dashboard)."""
    try:
        c = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"path": str(path), "loadable": False, "error": str(exc)}
    ok = isinstance(c, dict) and c.get("schema_version") == SCHEMA_VERSION
    return {"path": str(path), "loadable": True, "schema_version": c.get("schema_version") if isinstance(c, dict) else None,
            "kind": c.get("kind") if isinstance(c, dict) else None, "compatible": ok,
            "selection": c.get("selection") if ok else None, "progress": c.get("progress") if ok else None}
