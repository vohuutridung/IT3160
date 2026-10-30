"""Training script for Supervised Chess Policy ResNet-16x256.

Meets the specifications of Project Report N14:
- ResNet-16x256 backbone
- Policy head mapping to 4672 actions
- Legal move masking
- AdamW optimizer (lr=3e-4, weight_decay=1e-4)
- CosineAnnealingLR scheduler
- Effective batch size 512 (via microbatch + gradient accumulation)
- Mixed precision (AMP)
- Validation tracking with Top-1, Top-3, Top-5 accuracy and per-phase breakdowns
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Add src to pythonpath
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from IT3160.models.dataset import create_dataloader
from IT3160.models.metrics import MetricTracker, PHASE_NAMES
from IT3160.models.resnet import ChessPolicyResNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train Chess Policy ResNet-16x256")
    parser.add_argument("--train-path", type=str, default="data/tensors/official-v1/train.h5",
                        help="Path to training HDF5 file")
    parser.add_argument("--val-path", type=str, default="data/tensors/official-v1/validation.h5",
                        help="Path to validation HDF5 file")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Total number of epochs (default: 8)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Microbatch size per forward pass (default: 256)")
    parser.add_argument("--grad-accum-steps", type=int, default=2,
                        help="Gradient accumulation steps (256 * 2 = 512 effective batch)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Initial learning rate for AdamW (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay for AdamW (default: 1e-4)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="Optional limit on training samples (useful for smoke tests)")
    parser.add_argument("--limit-val", type=int, default=None,
                        help="Optional limit on validation samples (useful for smoke tests)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker count (default: 0)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on ('cuda', 'cpu', or auto-detect)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")
    return parser.parse_args()


def train_one_epoch(
    model: ChessPolicyResNet,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    grad_accum_steps: int,
    use_amp: bool,
    epoch: int,
) -> dict:
    model.train()
    tracker = MetricTracker()
    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    step_loss = 0.0

    for step, (x, y, legal_mask, phase) in enumerate(progress_bar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        legal_mask = legal_mask.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x, legal_mask=legal_mask)
                loss = F.cross_entropy(logits, y) / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            logits = model(x, legal_mask=legal_mask)
            loss = F.cross_entropy(logits, y) / grad_accum_steps
            loss.backward()

        unscaled_loss = loss.item() * grad_accum_steps
        step_loss += unscaled_loss
        tracker.update(logits, y, phase, unscaled_loss)

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 20 == 0:
            progress_bar.set_postfix({
                "loss": f"{unscaled_loss:.4f}",
                "top1": f"{tracker.top1_correct / max(1, tracker.total_samples) * 100:.1f}%",
            })

    return tracker.compute()


@torch.no_grad()
def evaluate(
    model: ChessPolicyResNet,
    dataloader,
    device: torch.device,
    desc: str = "Validation",
) -> dict:
    model.eval()
    tracker = MetricTracker()

    progress_bar = tqdm(dataloader, desc=desc, leave=False)
    for x, y, legal_mask, phase in progress_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        legal_mask = legal_mask.to(device, non_blocking=True)

        logits = model(x, legal_mask=legal_mask)
        loss = F.cross_entropy(logits, y).item()
        tracker.update(logits, y, phase, loss)

    return tracker.compute()


def print_metrics_summary(title: str, metrics: dict):
    print(f"\n--- {title} ---")
    print(f"  Samples: {metrics.get('samples', 0):,}")
    print(f"  Loss (NLL): {metrics['loss']:.4f}")
    print(f"  Top-1 Acc: {metrics['top1']:.2f}%")
    print(f"  Top-3 Acc: {metrics['top3']:.2f}%")
    print(f"  Top-5 Acc: {metrics['top5']:.2f}%")
    print("  Breakdown by phase:")
    for p, name in PHASE_NAMES.items():
        prefix = name.lower()
        if metrics.get(f"{prefix}_samples", 0) > 0:
            print(f"    - {name:10s} (N={metrics[f'{prefix}_samples']:,}): "
                  f"Top-1: {metrics[f'{prefix}_top1']:.2f}% | "
                  f"Top-3: {metrics[f'{prefix}_top3']:.2f}% | "
                  f"Top-5: {metrics[f'{prefix}_top5']:.2f}%")


def main():
    args = parse_args()
    print("=" * 60)
    print("IT3160 Chess Policy Training (ResNet-16x256)")
    print("=" * 60)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    use_amp = (device.type == "cuda") and (not args.no_amp)
    print(f"Automatic Mixed Precision (AMP): {use_amp}")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Create datasets & dataloaders
    print(f"\nLoading training data: {args.train_path}")
    train_loader = create_dataloader(
        args.train_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        limit=args.limit_train,
    )
    print(f"Loading validation data: {args.val_path}")
    val_loader = create_dataloader(
        args.val_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        limit=args.limit_val,
    )

    print(f"Train samples: {len(train_loader.dataset):,} | Batches: {len(train_loader)}")
    print(f"Val samples:   {len(val_loader.dataset):,} | Batches: {len(val_loader)}")

    # Model
    model = ChessPolicyResNet(in_channels=17, num_blocks=16, channels=256)
    model.to(device)
    total_params = model.count_parameters()
    print(f"Total model parameters: {total_params:,} (~{total_params * 4 / 1e6:.1f} MB in FP32)")

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume if requested
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        history = checkpoint.get("history", [])

    effective_batch = args.batch_size * args.grad_accum_steps
    print(f"\nEffective Batch Size: {effective_batch} (microbatch={args.batch_size}, accum={args.grad_accum_steps})")
    print(f"Learning rate: {args.lr} | Weight decay: {args.weight_decay}")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n>>> Epoch {epoch}/{args.epochs} (lr={current_lr:.2e}) <<<")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            use_amp=use_amp,
            epoch=epoch,
        )

        val_metrics = evaluate(model=model, dataloader=val_loader, device=device)
        scheduler.step()
        elapsed = time.time() - t0

        print_metrics_summary(f"Epoch {epoch} Training Results ({elapsed:.1f}s)", train_metrics)
        print_metrics_summary(f"Epoch {epoch} Validation Results", val_metrics)

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_seconds": elapsed,
        }
        history.append(epoch_record)

        # Save checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "config": vars(args),
        }
        torch.save(checkpoint_data, checkpoint_dir / "last_checkpoint.pt")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_data["best_val_loss"] = best_val_loss
            torch.save(checkpoint_data, checkpoint_dir / "best_model.pt")
            print(f"[*] New best validation loss: {best_val_loss:.4f} (Saved to best_model.pt)")

        # Save history log JSON
        with open(checkpoint_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training completed! Best Validation Loss: {best_val_loss:.4f}")
    print(f"Checkpoints and history saved to: {checkpoint_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
