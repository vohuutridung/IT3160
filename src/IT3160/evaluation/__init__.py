"""Evaluation framework for the chess policy models.

Core idea: run a model (or baseline) ONCE over an HDF5 split, save one row of
results per position (``predictions.parquet``), then do every analysis offline
from that file. Plain-language overview: ``docs/EVAL_OVERVIEW.md``.

Module map
----------
Running a policy
    checkpoint   load a ``.pt`` checkpoint; infers blocks/channels from the weights
    inference    run a policy over an HDF5 split -> ``Predictions`` (per-sample results)
    baselines    reference policies: random legal move, move-frequency prior
    meta         per-row FEN / move / Elo / game id, checked against the HDF5 labels
Analysing predictions
    metrics      NLL, Top-k, rank, calibration; overall and per phase
    slices       results by move type, piece, legal-move count, ply, Elo
    significance bootstrap confidence intervals (resampling whole games), McNemar
    runner       glue used by the scripts: evaluate one model, compare two
    report       Markdown tables
Playing games
    agent        pick a move for a ``chess.Board`` with a trained model
    arena        matches between players, colour-swapped openings, Elo estimate, PGN

Scripts: ``scripts/export_meta.py``, ``evaluate.py``, ``compare.py``, ``play_match.py``.
"""

from .inference import Predictions, load_predictions, run_inference, save_predictions
from .metrics import summarize, summarize_by_phase

__all__ = [
    "Predictions",
    "load_predictions",
    "run_inference",
    "save_predictions",
    "summarize",
    "summarize_by_phase",
]
