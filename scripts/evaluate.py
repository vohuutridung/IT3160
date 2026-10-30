"""Standalone evaluation script for Chess Policy ResNet on Test / Validation sets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add src to pythonpath
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from IT3160.models.dataset import create_dataloader
from IT3160.models.resnet import ChessPolicyResNet
from IT3160.models.metrics import MetricTracker, PHASE_NAMES
from train import evaluate, print_metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Chess Policy ResNet checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt model checkpoint (e.g. checkpoints/best_model.pt)")
    parser.add_argument("--test-path", type=str, default="data/tensors/official-v1/test.h5",
                        help="Path to test HDF5 file")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for evaluation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on test samples")
    parser.add_argument("--device", type=str, default=None,
                        help="Device ('cuda', 'cpu', or auto)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = ChessPolicyResNet(in_channels=17, num_blocks=16, channels=256)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loading test data: {args.test_path}")
    test_loader = create_dataloader(
        args.test_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        limit=args.limit,
    )
    print(f"Test samples: {len(test_loader.dataset):,} | Batches: {len(test_loader)}")

    metrics = evaluate(model, test_loader, device=device, desc="Testing")
    print_metrics_summary("Final Test Set Results", metrics)


if __name__ == "__main__":
    main()
