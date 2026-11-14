"""Run a policy once over an HDF5 split and keep per-sample results.

Every later analysis (metrics, slices, confidence intervals, model comparison)
reads these predictions, so the GPU pass happens only once per model and split.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Callable, Iterator

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

TOP_K = 5

# (x float32 [B,17,8,8], legal_mask bool [B,4672], phase long [B]) -> raw logits [B,4672]
PredictFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class Predictions:
    """Column arrays, one row per evaluated sample, in HDF5 row order."""

    row_idx: np.ndarray        # int64, row in the HDF5 file
    y: np.ndarray              # int64, played action
    phase: np.ndarray          # uint8, 0 opening / 1 middlegame / 2 endgame
    n_legal: np.ndarray        # int16, number of legal moves
    logp_y: np.ndarray         # float32, log-probability of the played move (masked softmax)
    rank_y: np.ndarray         # int32, number of legal moves scored >= the played move (ties count against it)
    top_ids: np.ndarray        # int32 [N, 5], best actions (masked)
    top1_logp: np.ndarray      # float32, log-probability of the best action
    entropy: np.ndarray        # float32, entropy of the masked distribution (nats)
    unmasked_top1_legal: np.ndarray  # bool, argmax of the RAW logits is a legal move

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, index: np.ndarray) -> "Predictions":
        return Predictions(**{f.name: getattr(self, f.name)[index] for f in fields(self)})


def iter_h5_batches(
    h5_path: str | Path,
    batch_size: int = 1024,
    limit: int | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Sequential contiguous reads: one chunk decompression per chunk, not per sample."""
    with h5py.File(h5_path, "r") as source:
        total = source["y"].shape[0]
        n = total if limit is None or limit <= 0 else min(total, limit)
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            yield (
                start,
                source["x"][start:stop],
                source["y"][start:stop],
                source["legal_mask"][start:stop],
                source["phase"][start:stop],
            )


def h5_length(h5_path: str | Path, limit: int | None = None) -> int:
    with h5py.File(h5_path, "r") as source:
        total = source["y"].shape[0]
    return total if limit is None or limit <= 0 else min(total, limit)


def model_predict_fn(model: torch.nn.Module, device: torch.device, use_amp: bool = False) -> PredictFn:
    """Wrap a ChessPolicyResNet; returns its unmasked logits in float32.

    fp32 by default so a model scores identically on CPU and GPU; fp16 autocast
    (``use_amp``) is faster but can create ties and shift ranks.
    """
    amp = use_amp and device.type == "cuda"

    def predict(x: torch.Tensor, legal_mask: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        return logits.float()

    return predict


@torch.no_grad()
def score_batch(
    logits: torch.Tensor,
    y: torch.Tensor,
    legal_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-sample statistics of the legal-masked distribution."""
    logits = logits.float()
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    logp = F.log_softmax(masked, dim=-1)
    logp_y = logp.gather(1, y.unsqueeze(1)).squeeze(1)
    # Pessimistic rank: legal moves tied with the label count as ranked above it,
    # so ties (count-based priors, low-precision logits) never inflate Top-k.
    rank_y = (logp >= logp_y.unsqueeze(1)).sum(dim=1)
    top_logp, top_ids = logp.topk(TOP_K, dim=-1)
    plogp = torch.where(legal_mask, logp.exp() * logp, torch.zeros_like(logp))
    unmasked_top1 = logits.argmax(dim=-1)
    return {
        "logp_y": logp_y,
        "rank_y": rank_y,
        "top_ids": top_ids,
        "top1_logp": top_logp[:, 0],
        "entropy": -plogp.sum(dim=-1),
        "unmasked_top1_legal": legal_mask.gather(1, unmasked_top1.unsqueeze(1)).squeeze(1),
        "n_legal": legal_mask.sum(dim=-1),
    }


@torch.no_grad()
def run_inference(
    predict_fn: PredictFn,
    h5_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 1024,
    limit: int | None = None,
    progress: bool = True,
) -> Predictions:
    device = torch.device(device)
    n = h5_length(h5_path, limit)
    out = {
        "row_idx": np.arange(n, dtype=np.int64),
        "y": np.empty(n, np.int64),
        "phase": np.empty(n, np.uint8),
        "n_legal": np.empty(n, np.int16),
        "logp_y": np.empty(n, np.float32),
        "rank_y": np.empty(n, np.int32),
        "top_ids": np.empty((n, TOP_K), np.int32),
        "top1_logp": np.empty(n, np.float32),
        "entropy": np.empty(n, np.float32),
        "unmasked_top1_legal": np.empty(n, np.bool_),
    }
    batches = iter_h5_batches(h5_path, batch_size=batch_size, limit=limit)
    if progress:
        from tqdm import tqdm
        batches = tqdm(batches, total=-(-n // batch_size), desc="Inference", leave=False)
    for start, x, y, mask, phase in batches:
        stop = start + len(y)
        y_np = y.astype(np.int64)
        if not mask[np.arange(len(y_np)), y_np].all():
            raise ValueError(f"Target outside legal mask in rows {start}..{stop - 1}")
        x_t = torch.from_numpy(x).to(device, non_blocking=True).float()
        y_t = torch.from_numpy(y_np).to(device)
        mask_t = torch.from_numpy(mask).to(device).bool()
        phase_t = torch.from_numpy(phase.astype(np.int64)).to(device)
        stats = score_batch(predict_fn(x_t, mask_t, phase_t), y_t, mask_t)
        out["y"][start:stop] = y_np
        out["phase"][start:stop] = phase
        for key, value in stats.items():
            out[key][start:stop] = value.cpu().numpy()
    return Predictions(**out)


def save_predictions(predictions: Predictions, path: str | Path, metadata: dict | None = None) -> None:
    columns = {
        f.name: getattr(predictions, f.name) for f in fields(predictions) if f.name != "top_ids"
    }
    for k in range(TOP_K):
        columns[f"top{k + 1}_id"] = predictions.top_ids[:, k]
    table = pa.Table.from_pydict(columns)
    if metadata:
        table = table.replace_schema_metadata({b"it3160.eval": json.dumps(metadata).encode()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def load_predictions(path: str | Path) -> Predictions:
    table = pq.read_table(path)
    data = {name: table.column(name).to_numpy() for name in table.column_names}
    top_ids = np.stack([data.pop(f"top{k + 1}_id") for k in range(TOP_K)], axis=1)
    return Predictions(top_ids=top_ids, **data)


def load_prediction_metadata(path: str | Path) -> dict:
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(b"it3160.eval")
    return json.loads(raw) if raw else {}
