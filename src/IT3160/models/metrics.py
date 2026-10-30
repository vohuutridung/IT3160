"""Evaluation metrics for chess policy training and testing."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

PHASE_NAMES = {
    0: "Opening",
    1: "Middlegame",
    2: "Endgame",
}


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Compute Top-k accuracy given logits and 1D target action IDs."""
    if logits.size(0) == 0:
        return 0.0
    k = min(k, logits.size(-1))
    topk_preds = logits.topk(k, dim=-1).indices  # [B, k]
    correct = topk_preds.eq(targets.unsqueeze(-1)).any(dim=-1).float().sum().item()
    return correct / logits.size(0)


class MetricTracker:
    """Tracks overall and per-phase metrics across evaluation batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.total_loss = 0.0
        self.top1_correct = 0
        self.top3_correct = 0
        self.top5_correct = 0

        # Phase statistics: {phase_id: {"total": int, "top1": int, "top3": int, "top5": int, "loss": float}}
        self.phase_stats: Dict[int, Dict[str, float]] = {
            p: {"total": 0, "top1": 0, "top3": 0, "top5": 0, "loss": 0.0}
            for p in (0, 1, 2)
        }

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        phases: torch.Tensor,
        loss: float,
    ) -> None:
        batch_size = targets.size(0)
        if batch_size == 0:
            return

        self.total_samples += batch_size
        self.total_loss += loss * batch_size

        # Top-k predictions
        top5_preds = logits.topk(5, dim=-1).indices  # [B, 5]
        targets_expanded = targets.unsqueeze(-1)      # [B, 1]

        c1 = top5_preds[:, :1].eq(targets_expanded).any(dim=-1)
        c3 = top5_preds[:, :3].eq(targets_expanded).any(dim=-1)
        c5 = top5_preds[:, :5].eq(targets_expanded).any(dim=-1)

        self.top1_correct += c1.sum().item()
        self.top3_correct += c3.sum().item()
        self.top5_correct += c5.sum().item()

        # Update per-phase
        phases_cpu = phases.detach().cpu()
        c1_cpu = c1.detach().cpu()
        c3_cpu = c3.detach().cpu()
        c5_cpu = c5.detach().cpu()

        for p in (0, 1, 2):
            mask = (phases_cpu == p)
            cnt = mask.sum().item()
            if cnt > 0:
                self.phase_stats[p]["total"] += cnt
                self.phase_stats[p]["top1"] += c1_cpu[mask].sum().item()
                self.phase_stats[p]["top3"] += c3_cpu[mask].sum().item()
                self.phase_stats[p]["top5"] += c5_cpu[mask].sum().item()
                self.phase_stats[p]["loss"] += loss * cnt

    def compute(self) -> Dict[str, float]:
        if self.total_samples == 0:
            return {
                "loss": 0.0,
                "top1": 0.0,
                "top3": 0.0,
                "top5": 0.0,
            }

        n = self.total_samples
        result = {
            "loss": self.total_loss / n,
            "top1": (self.top1_correct / n) * 100.0,
            "top3": (self.top3_correct / n) * 100.0,
            "top5": (self.top5_correct / n) * 100.0,
            "samples": n,
        }

        for p, name in PHASE_NAMES.items():
            ptotal = self.phase_stats[p]["total"]
            prefix = name.lower()
            if ptotal > 0:
                result[f"{prefix}_top1"] = (self.phase_stats[p]["top1"] / ptotal) * 100.0
                result[f"{prefix}_top3"] = (self.phase_stats[p]["top3"] / ptotal) * 100.0
                result[f"{prefix}_top5"] = (self.phase_stats[p]["top5"] / ptotal) * 100.0
                result[f"{prefix}_samples"] = ptotal
            else:
                result[f"{prefix}_top1"] = 0.0
                result[f"{prefix}_top3"] = 0.0
                result[f"{prefix}_top5"] = 0.0
                result[f"{prefix}_samples"] = 0

        return result


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float, float]:
    """Quick computation of Top-1, Top-3, Top-5 accuracy percentages."""
    return (
        topk_accuracy(logits, targets, k=1) * 100.0,
        topk_accuracy(logits, targets, k=3) * 100.0,
        topk_accuracy(logits, targets, k=5) * 100.0,
    )
