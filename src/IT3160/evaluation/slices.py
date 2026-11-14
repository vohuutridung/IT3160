"""Break metrics down by move type, position complexity, ply and rating."""

from __future__ import annotations

import chess
import numpy as np

from IT3160.data.action_encoding import decode_move

from .baselines import random_legal_metrics
from .inference import Predictions
from .metrics import PHASE_NAMES, summarize

SLICE_METRICS = ("samples", "nll", "top1", "top3", "top5", "mean_rank")
# Natural order of bin labels; anything else is sorted alphabetically after them.
BIN_ORDER = (
    "<=10", "11-25", "26-40", ">40",
    "0-24", "25-60", "61-100", ">100",
    "<2400", "2400-2599", "2600-2799", ">=2800",
    "unknown",
)


def _label_key(label: str) -> tuple[int, str]:
    return (BIN_ORDER.index(label) if label in BIN_ORDER else len(BIN_ORDER), label)


def _bin(values: np.ndarray, edges: list[float], labels: list[str]) -> np.ndarray:
    """Assign ``labels[i]`` when ``edges[i-1] < v <= edges[i]``; NaN -> 'unknown'."""
    values = np.asarray(values, np.float64)
    out = np.full(len(values), "unknown", dtype=object)
    known = ~np.isnan(values)
    out[known] = np.asarray(labels, dtype=object)[np.searchsorted(edges, values[known], side="left")]
    return out


def legal_moves_bins(n_legal: np.ndarray) -> np.ndarray:
    return _bin(n_legal, [10, 25, 40], ["<=10", "11-25", "26-40", ">40"])


def ply_bins(ply: np.ndarray) -> np.ndarray:
    return _bin(ply, [24, 60, 100], ["0-24", "25-60", "61-100", ">100"])


def elo_bins(white_elo: np.ndarray, black_elo: np.ndarray) -> np.ndarray:
    mean = (np.asarray(white_elo, np.float64) + np.asarray(black_elo, np.float64)) / 2
    return _bin(mean, [2399, 2599, 2799], ["<2400", "2400-2599", "2600-2799", ">=2800"])


def move_features(fens: np.ndarray, moves: np.ndarray) -> dict[str, np.ndarray]:
    """Kind of the played move (castling > promotion > capture > quiet), piece, check."""
    kind = np.empty(len(fens), dtype=object)
    piece = np.empty(len(fens), dtype=object)
    check = np.empty(len(fens), dtype=object)
    for i, (fen, uci) in enumerate(zip(fens, moves)):
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if board.is_castling(move):
            kind[i] = "castling"
        elif move.promotion == chess.QUEEN:
            kind[i] = "promotion"
        elif move.promotion is not None:
            kind[i] = "underpromotion"
        elif board.is_capture(move):
            kind[i] = "capture"
        else:
            kind[i] = "quiet"
        piece[i] = chess.piece_name(board.piece_type_at(move.from_square))
        check[i] = "check" if board.gives_check(move) else "no_check"
    return {"move_kind": kind, "piece": piece, "gives_check": check}


def compute_slice_labels(predictions: Predictions, meta: dict[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    """All slice label arrays available; move/ply/Elo slices need metadata."""
    labels = {"legal_moves": legal_moves_bins(predictions.n_legal)}
    if meta is not None:
        labels.update(move_features(meta["fen"], meta["move_uci"]))
        labels["ply"] = ply_bins(meta["ply"])
        labels["elo"] = elo_bins(meta["white_elo"], meta["black_elo"])
    return labels


def slice_summary(predictions: Predictions, labels: np.ndarray) -> dict[str, dict]:
    result = {}
    for value in sorted(set(labels.tolist()), key=_label_key):
        subset = predictions.subset(labels == value)
        summary = summarize(subset)
        result[value] = {key: summary[key] for key in SLICE_METRICS if key in summary}
        if len(subset):  # random-move reference on the same positions
            random = random_legal_metrics(subset.n_legal)
            result[value].update(random_nll=random["nll"], random_top1=random["top1"])
    return result


def all_slices(predictions: Predictions, meta: dict[str, np.ndarray] | None) -> dict:
    return {
        name: slice_summary(predictions, labels)
        for name, labels in compute_slice_labels(predictions, meta).items()
    }


def worst_examples(
    predictions: Predictions,
    meta: dict[str, np.ndarray],
    per_phase: int = 5,
) -> list[dict]:
    """Samples whose played move received the lowest probability, per phase."""
    rows = []
    for phase_id in range(len(PHASE_NAMES)):
        index = np.flatnonzero(predictions.phase == phase_id)
        order = index[np.argsort(predictions.logp_y[index])[:per_phase]]
        for i in order:
            board = chess.Board(meta["fen"][i])
            predicted = []
            for action in predictions.top_ids[i][:3]:
                try:
                    predicted.append(decode_move(board, int(action)).uci())
                except ValueError:
                    break
            rows.append({
                "row_idx": int(predictions.row_idx[i]), "phase": int(phase_id),
                "fen": meta["fen"][i], "played": meta["move_uci"][i],
                "p_played": float(np.exp(predictions.logp_y[i])),
                "rank_played": int(predictions.rank_y[i]), "model_top3": predicted,
            })
    return rows
