"""Aggregate metrics computed from per-sample predictions."""

from __future__ import annotations

import numpy as np

from .inference import Predictions

PHASE_NAMES = ("opening", "middlegame", "endgame")
GROUPS = ("overall", *PHASE_NAMES)
TOPK = (1, 3, 5)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Standard ECE over equal-width confidence bins (returned in [0, 1])."""
    if len(confidence) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        selected = index == b
        if selected.any():
            total += selected.sum() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(total / len(confidence))


def reliability_table(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for b in range(bins):
        selected = index == b
        count = int(selected.sum())
        rows.append({
            "bin_low": float(edges[b]), "bin_high": float(edges[b + 1]), "count": count,
            "confidence": float(confidence[selected].mean()) if count else None,
            "accuracy": float(correct[selected].mean()) if count else None,
        })
    return rows


def summarize(predictions: Predictions) -> dict:
    """Headline metrics. Accuracies and rates are percentages, NLL in nats.

    nll                    mean -log p(played move); lower = less "surprised"
    topk                   % of positions where the played move is among the k best
    mrr                    mean of 1/rank of the played move
    ece                    calibration error: |confidence - accuracy| averaged over 10 bins
    unmasked_illegal_top1  % of positions where the RAW (unmasked) best move is illegal
    """
    n = len(predictions)
    if n == 0:
        return {"samples": 0}
    rank = predictions.rank_y.astype(np.float64)
    confidence = np.exp(predictions.top1_logp.astype(np.float64))
    correct = predictions.rank_y == 1
    result = {
        "samples": int(n),
        "nll": float(-predictions.logp_y.astype(np.float64).mean()),
    }
    for k in TOPK:
        result[f"top{k}"] = float((predictions.rank_y <= k).mean() * 100.0)
    result.update({
        "mean_rank": float(rank.mean()),
        "median_rank": float(np.median(rank)),
        "mrr": float((1.0 / rank).mean()),
        "mean_entropy": float(predictions.entropy.astype(np.float64).mean()),
        "mean_confidence": float(confidence.mean() * 100.0),
        "ece": float(expected_calibration_error(confidence, correct) * 100.0),
        "unmasked_illegal_top1": float((~predictions.unmasked_top1_legal).mean() * 100.0),
        "mean_legal_moves": float(predictions.n_legal.astype(np.float64).mean()),
    })
    return result


def summarize_by_phase(predictions: Predictions) -> dict:
    """``{"overall": {...}, "opening": {...}, "middlegame": {...}, "endgame": {...}}``."""
    result = {"overall": summarize(predictions)}
    for phase_id, name in enumerate(PHASE_NAMES):
        result[name] = summarize(predictions.subset(predictions.phase == phase_id))
    return result


def per_sample_values(predictions: Predictions) -> dict[str, np.ndarray]:
    """Per-sample quantities whose mean is a headline metric (used for bootstrap)."""
    values = {"nll": -predictions.logp_y.astype(np.float64)}
    for k in TOPK:
        values[f"top{k}"] = (predictions.rank_y <= k).astype(np.float64) * 100.0
    return values
