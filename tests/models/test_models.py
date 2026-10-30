"""Unit tests for models, dataset loader, and metrics."""

from pathlib import Path
import pytest
import torch

from IT3160.models.dataset import ChessTensorDataset
from IT3160.models.metrics import MetricTracker, topk_accuracy
from IT3160.models.resnet import ChessPolicyResNet

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "tensors" / "official-v1"


def test_resnet_forward():
    model = ChessPolicyResNet(in_channels=17, num_blocks=2, channels=32)
    batch_size = 4
    x = torch.randn(batch_size, 17, 8, 8)
    legal_mask = torch.ones(batch_size, 4672, dtype=torch.bool)
    legal_mask[:, 100:] = False  # Only 0..99 legal

    logits = model(x, legal_mask=legal_mask)
    assert logits.shape == (batch_size, 4672)
    assert (logits[:, 100:] <= -1e8).all()


def test_metrics_tracker():
    tracker = MetricTracker()
    # Batch with 2 samples
    logits = torch.zeros(2, 4672)
    # Sample 0: correct prediction is 10
    logits[0, 10] = 5.0
    # Sample 1: top prediction is 20, correct is 21 (rank 2)
    logits[1, 20] = 5.0
    logits[1, 21] = 4.0

    targets = torch.tensor([10, 21], dtype=torch.long)
    phases = torch.tensor([0, 1], dtype=torch.long)  # opening, middlegame

    tracker.update(logits, targets, phases, loss=1.5)
    res = tracker.compute()

    assert res["samples"] == 2
    assert res["top1"] == 50.0  # 1 out of 2 correct
    assert res["top3"] == 100.0  # both in top 3
    assert res["opening_top1"] == 100.0
    assert res["middlegame_top1"] == 0.0


def test_dataset_loader():
    val_path = DATA_DIR / "validation.h5"
    if not val_path.is_file():
        pytest.skip("validation.h5 not found locally")

    ds = ChessTensorDataset(val_path, limit=5)
    assert len(ds) == 5
    x, y, mask, phase = ds[0]

    assert x.shape == (17, 8, 8)
    assert x.dtype == torch.float32
    assert y.ndim == 0
    assert mask.shape == (4672,)
    assert mask.dtype == torch.bool
    assert mask[y] == True  # target must be a legal move!
    assert phase.item() in (0, 1, 2)
