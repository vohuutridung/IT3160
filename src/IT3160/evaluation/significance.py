"""Confidence intervals and paired tests for comparing policies on one test set.

Bootstrap resamples whole groups (games) when group labels are available, since
positions from the same game are correlated; otherwise it resamples samples.
"""

from __future__ import annotations

import math

import numpy as np


def _group_sums(values: np.ndarray, groups: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if groups is None:
        return values.astype(np.float64), np.ones(len(values), np.float64)
    _, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values.astype(np.float64))
    counts = np.bincount(inverse).astype(np.float64)
    return sums, counts


def bootstrap_mean_ci(
    values: np.ndarray,
    groups: np.ndarray | None = None,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    chunk: int = 50,
) -> dict:
    """Percentile CI of the per-sample mean, resampling groups with replacement."""
    values = np.asarray(values, np.float64)
    estimate = float(values.mean()) if len(values) else float("nan")
    if len(values) == 0 or n_boot <= 0:
        return {"estimate": estimate, "low": None, "high": None}
    sums, counts = _group_sums(values, groups)
    g = len(sums)
    rng = np.random.default_rng(seed)
    stats = []
    for start in range(0, n_boot, chunk):
        reps = min(chunk, n_boot - start)
        draws = rng.integers(0, g, size=(reps, g))
        weights = np.stack([np.bincount(row, minlength=g) for row in draws]).astype(np.float64)
        stats.append((weights @ sums) / (weights @ counts))
    stats = np.concatenate(stats)
    return {
        "estimate": estimate,
        "low": float(np.quantile(stats, alpha / 2)),
        "high": float(np.quantile(stats, 1 - alpha / 2)),
    }


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    groups: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """CI of mean(a - b) on the same samples; an interval excluding 0 is significant."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have the same shape")
    return bootstrap_mean_ci(a - b, groups, **kwargs)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray, exact_limit: int = 1000) -> dict:
    """McNemar test on paired binary outcomes (e.g. Top-1 hits of two models).

    Exact two-sided binomial test when the discordant count is small, otherwise
    the chi-square statistic with continuity correction.
    """
    correct_a = np.asarray(correct_a, bool)
    correct_b = np.asarray(correct_b, bool)
    if correct_a.shape != correct_b.shape:
        raise ValueError("Paired arrays must have the same shape")
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    discordant = only_a + only_b
    if discordant == 0:
        p_value, method, statistic = 1.0, "exact", 0.0
    elif discordant <= exact_limit:
        k = min(only_a, only_b)
        log_terms = [
            math.lgamma(discordant + 1) - math.lgamma(i + 1) - math.lgamma(discordant - i + 1)
            - discordant * math.log(2.0)
            for i in range(k + 1)
        ]
        peak = max(log_terms)
        tail = math.exp(peak) * sum(math.exp(t - peak) for t in log_terms)
        p_value, method, statistic = min(1.0, 2.0 * tail), "exact", float(k)
    else:
        statistic = (abs(only_a - only_b) - 1) ** 2 / discordant
        p_value, method = math.erfc(math.sqrt(statistic / 2.0)), "chi2_cc"
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "method": method,
    }
