"""Reference policies that give the model's numbers a meaningful floor."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from IT3160.data.schema import ACTION_SPACE_SIZE

from .inference import PredictFn, h5_length, iter_h5_batches
from .metrics import PHASE_NAMES, TOPK


def random_legal_summary(h5_path: str | Path, limit: int | None = None) -> dict:
    """Exact expected metrics of a uniformly random legal move (no sampling noise)."""
    n_legal = np.empty(h5_length(h5_path, limit), np.float64)
    phase = np.empty_like(n_legal, dtype=np.uint8)
    for start, _x, y, mask, ph in iter_h5_batches(h5_path, limit=limit):
        n_legal[start:start + len(y)] = mask.sum(axis=1)
        phase[start:start + len(y)] = ph

    result = {"overall": random_legal_metrics(n_legal)}
    for phase_id, name in enumerate(PHASE_NAMES):
        result[name] = random_legal_metrics(n_legal[phase == phase_id])
    return result


def random_legal_metrics(n_legal: np.ndarray) -> dict:
    """Expected metrics of a uniformly random legal move, given each position's legal-move count."""
    n = np.asarray(n_legal, np.float64)
    if len(n) == 0:
        return {"samples": 0}
    result = {"samples": int(len(n)), "nll": float(np.log(n).mean())}
    for k in TOPK:
        result[f"top{k}"] = float((np.minimum(k, n) / n).mean() * 100.0)
    result.update({
        "mean_rank": float(((n + 1) / 2).mean()),
        "mean_entropy": result["nll"],  # uniform distribution: entropy == NLL
        "mean_legal_moves": float(n.mean()),
    })
    return result


def fit_action_prior(
    train_h5: str | Path,
    alpha: float = 1.0,
    chunk: int = 1 << 18,
) -> np.ndarray:
    """Log-frequency of each action per phase on the training split, shape [3, 4672]."""
    counts = np.zeros((len(PHASE_NAMES), ACTION_SPACE_SIZE), np.float64)
    with h5py.File(train_h5, "r") as source:
        n = source["y"].shape[0]
        for start in range(0, n, chunk):
            y = source["y"][start:start + chunk].astype(np.int64)
            phase = source["phase"][start:start + chunk].astype(np.int64)
            np.add.at(counts, (phase, y), 1.0)
    counts += alpha
    return np.log(counts / counts.sum(axis=1, keepdims=True)).astype(np.float32)


def prior_predict_fn(log_prior: np.ndarray, device: torch.device | str = "cpu") -> PredictFn:
    """Score each legal move by how often it was played in that phase during training."""
    table = torch.from_numpy(log_prior).to(device)

    def predict(x: torch.Tensor, legal_mask: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return table[phase]

    return predict
