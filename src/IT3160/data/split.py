"""
Deterministically assign chess games to dataset splits.
- Validate train/validation/test ratios.
- Assign each game to exactly one split.
- Ensure reproducibility across runs and processing orders.
"""

import hashlib
import json
import math

from dataclasses import dataclass

from .schema import Split


DEFAULT_SEED = 42
SPLIT_HASH_VERSION = "sha256-v1"


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Dataset split probabilities."""

    train: float = 0.90
    validation: float = 0.05
    test: float = 0.05

    def __post_init__(self):
        values = (
            self.train,
            self.validation,
            self.test,
        )

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in values
        ):
            raise ValueError(
                "All split ratios must be positive, finite numbers"
            )

        if not math.isclose(
            sum(values),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Split ratios must sum to 1.0"
            )


DEFAULT_RATIOS = SplitRatios()


def _hash_fraction(game_id: str, seed: int) -> float:
    """
    Deterministically map (seed, game_id) into [0, 1).
    Uses SHA-256 instead of Python's built-in hash(), whose output is not stable across interpreter runs.
    """

    payload = json.dumps(
        {
            "version": SPLIT_HASH_VERSION,
            "seed": seed,
            "game_id": game_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    # Convert the first 8 bytes to an unsigned 64-bit integer.
    value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    # The largest uint64 values round to 1.0 as binary64 floats. Keep the
    # documented half-open interval while preserving existing hash mappings.
    return min(value / (1 << 64), math.nextafter(1.0, 0.0))


def assign_split(
    game_id: str,
    seed: int = DEFAULT_SEED,
    ratios: SplitRatios = DEFAULT_RATIOS,
) -> Split:
    """
    Assign a chess game to a deterministic dataset split.
    The result depends only on: game_id, seed, split ratios
    Returns:
        Split.TRAIN, Split.VALIDATION, or Split.TEST.
    """

    if not isinstance(game_id, str) or not game_id.strip():
        raise ValueError(
            "game_id must be a non-empty string"
        )

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if not isinstance(ratios, SplitRatios):
        raise TypeError("ratios must be a SplitRatios instance")

    value = _hash_fraction(game_id, seed)

    train_threshold = ratios.train
    validation_threshold = ratios.train + ratios.validation

    if value < train_threshold:
        return Split.TRAIN

    if value < validation_threshold:
        return Split.VALIDATION

    return Split.TEST
