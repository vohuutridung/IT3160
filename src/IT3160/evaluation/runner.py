"""High-level evaluation and comparison used by ``scripts/evaluate.py`` and ``scripts/compare.py``."""

from __future__ import annotations

import numpy as np

from .inference import Predictions
from .metrics import GROUPS, PHASE_NAMES, per_sample_values, reliability_table, summarize_by_phase
from .significance import bootstrap_mean_ci, mcnemar_test, paired_bootstrap_diff
from .slices import all_slices, worst_examples


def _group_index(predictions: Predictions, group: str) -> np.ndarray:
    if group == "overall":
        return np.arange(len(predictions))
    return np.flatnonzero(predictions.phase == PHASE_NAMES.index(group))


def evaluate_predictions(
    predictions: Predictions,
    meta: dict[str, np.ndarray] | None = None,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Headline metrics per phase, bootstrap CIs, calibration and slices."""
    result = summarize_by_phase(predictions)
    games = meta["game_id"] if meta is not None else None
    result["ci"] = {}
    for group in GROUPS:
        index = _group_index(predictions, group)
        values = per_sample_values(predictions.subset(index))
        groups = games[index] if games is not None else None
        result["ci"][group] = {
            key: bootstrap_mean_ci(v, groups, n_boot=n_boot, seed=seed) for key, v in values.items()
        }
    result["ci_resampling"] = "game" if games is not None else "sample"
    result["reliability"] = reliability_table(
        np.exp(predictions.top1_logp.astype(np.float64)), predictions.rank_y == 1,
    )
    result["slices"] = all_slices(predictions, meta)
    if meta is not None:
        result["worst_examples"] = worst_examples(predictions, meta)
    return result


def check_aligned(reference: Predictions, other: Predictions) -> None:
    if not (np.array_equal(reference.row_idx, other.row_idx) and np.array_equal(reference.y, other.y)):
        raise ValueError("Predictions are not on the same samples (row_idx / y differ)")


def paired_comparison(
    reference_name: str,
    reference: Predictions,
    name: str,
    other: Predictions,
    meta: dict[str, np.ndarray] | None = None,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[dict]:
    """Per group: Δ = other − reference for Top-1 (pp) and NLL, with McNemar on Top-1."""
    check_aligned(reference, other)
    rows = []
    for group in GROUPS:
        index = _group_index(reference, group)
        groups = meta["game_id"][index] if meta is not None else None
        ref_hit = reference.rank_y[index] == 1
        hit = other.rank_y[index] == 1
        rows.append({
            "reference": reference_name, "model": name, "group": group,
            "delta_top1": paired_bootstrap_diff(
                hit * 100.0, ref_hit * 100.0, groups, n_boot=n_boot, seed=seed),
            "delta_nll": paired_bootstrap_diff(
                -other.logp_y[index], -reference.logp_y[index], groups, n_boot=n_boot, seed=seed),
            "mcnemar": mcnemar_test(hit, ref_hit),
        })
    return rows
