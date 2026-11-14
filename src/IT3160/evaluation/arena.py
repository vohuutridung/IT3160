"""Controlled matches between two players.

Protocol (from the proposal): the same set of opening positions for both
players, every opening played twice with colours swapped, one model forward
pass per move (no search), standard draw rules plus a ply cap. The result is
W/D/L from player A's point of view, an Elo difference estimate with a 95% CI
computed over opening pairs, and every game as PGN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Iterable, Protocol

import chess
import chess.pgn
import numpy as np

from .agent import PolicyAgent

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


class Player(Protocol):
    name: str

    def ranked_moves(self, board: chess.Board, rng: np.random.Generator) -> list[tuple[chess.Move, float]]:
        """Legal moves with non-negative weights, best first."""


class ModelPlayer:
    """Single forward pass of a policy network."""

    def __init__(self, name: str, agent: PolicyAgent):
        self.name = name
        self.agent = agent

    def ranked_moves(self, board, rng):
        return self.agent.policy(board)


class RandomPlayer:
    """Uniformly random legal move (sanity-check opponent)."""

    def __init__(self, name: str = "random"):
        self.name = name

    def ranked_moves(self, board, rng):
        moves = list(board.legal_moves)
        rng.shuffle(moves)
        return [(move, 1.0 / len(moves)) for move in moves]


def _material(board: chess.Board, color: bool) -> int:
    return sum(
        value * (len(board.pieces(piece, color)) - len(board.pieces(piece, not color)))
        for piece, value in PIECE_VALUES.items()
    )


class GreedyMaterialPlayer:
    """One-ply greedy: mate if possible, else maximise material, random tie-break."""

    def __init__(self, name: str = "greedy"):
        self.name = name

    def ranked_moves(self, board, rng):
        me = board.turn
        scored = []
        for move in board.legal_moves:
            board.push(move)
            score = 1000 if board.is_checkmate() else _material(board, me)
            board.pop()
            scored.append((score, rng.random(), move))
        scored.sort(key=lambda item: (-item[0], item[1]))
        best = scored[0][0]
        return [(move, 1.0 if score == best else 0.0) for score, _, move in scored]


def select_move(
    player: Player,
    board: chess.Board,
    rng: np.random.Generator,
    *,
    temperature: float = 0.0,
    avoid_repetition: bool = True,
) -> chess.Move:
    """Best (or sampled) move; optionally skip moves allowing a threefold-repetition claim."""
    ranked = player.ranked_moves(board, rng)
    if not ranked:
        raise ValueError("No legal moves")
    moves = [move for move, _ in ranked]
    weights = np.array([max(w, 0.0) for _, w in ranked], np.float64)
    if temperature > 0 and weights.sum() > 0:
        weights = weights ** (1.0 / temperature)
        # Weighted random permutation (Efraimidis–Spirakis): the first item is a sample
        # from p ∝ weights, later items are fallbacks if repetition is avoided.
        with np.errstate(divide="ignore"):
            keys = np.where(weights > 0, rng.random(len(moves)) ** (1.0 / weights), -1.0)
        order = np.argsort(-keys, kind="stable")
        moves = [moves[i] for i in order]
    if avoid_repetition:
        for move in moves:
            board.push(move)
            repeats = board.can_claim_threefold_repetition()
            board.pop()
            if not repeats:
                return move
    return moves[0]


@dataclass
class GameRecord:
    opening_index: int
    opening_fen: str
    white: str
    black: str
    result: str              # "1-0", "0-1", "1/2-1/2"
    termination: str         # checkmate, stalemate, threefold_repetition, ..., max_plies
    plies: int
    moves: list[str] = field(default_factory=list)

    def score_for(self, name: str) -> float:
        if self.result == "1/2-1/2":
            return 0.5
        winner = self.white if self.result == "1-0" else self.black
        return 1.0 if winner == name else 0.0


def play_game(
    white: Player,
    black: Player,
    opening_fen: str,
    rng: np.random.Generator,
    *,
    opening_index: int = 0,
    max_plies: int = 300,
    temperature: float = 0.0,
    avoid_repetition: bool = True,
) -> GameRecord:
    board = chess.Board(opening_fen)
    moves = []
    termination, result = "max_plies", "1/2-1/2"
    while len(moves) < max_plies:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            termination = outcome.termination.name.lower()
            result = outcome.result()
            break
        player = white if board.turn == chess.WHITE else black
        move = select_move(player, board, rng, temperature=temperature,
                           avoid_repetition=avoid_repetition)
        if not board.is_legal(move):  # defensive: agents must never return illegal moves
            raise RuntimeError(f"{player.name} played an illegal move {move.uci()}")
        moves.append(move.uci())
        board.push(move)
    else:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            termination, result = outcome.termination.name.lower(), outcome.result()
    return GameRecord(opening_index, opening_fen, white.name, black.name, result,
                      termination, len(moves), moves)


def play_match(
    player_a: Player,
    player_b: Player,
    openings: Iterable[str],
    *,
    seed: int = 0,
    max_plies: int = 300,
    temperature: float = 0.0,
    avoid_repetition: bool = True,
    progress: bool = True,
) -> list[GameRecord]:
    """Each opening twice: A as White, then B as White. Deterministic for a seed."""
    openings = list(openings)
    if player_a.name == player_b.name:
        raise ValueError("Players need distinct names")
    games = []
    iterator = enumerate(openings)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(openings), desc="Openings", leave=False)
    for index, fen in iterator:
        for game_number, (white, black) in enumerate(((player_a, player_b), (player_b, player_a))):
            rng = np.random.default_rng([seed, index, game_number])
            games.append(play_game(white, black, fen, rng, opening_index=index,
                                   max_plies=max_plies, temperature=temperature,
                                   avoid_repetition=avoid_repetition))
    return games


def _elo(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


def summarize_match(games: list[GameRecord], name_a: str, name_b: str) -> dict:
    """W/D/L for A, Elo difference (A − B) with a 95% CI over opening pairs."""
    scores = np.array([game.score_for(name_a) for game in games], np.float64)
    wins = int((scores == 1).sum())
    draws = int((scores == 0.5).sum())
    losses = int((scores == 0).sum())
    n = len(games)
    mean = float(scores.mean()) if n else float("nan")
    pairs: dict[int, list[float]] = {}
    for game, score in zip(games, scores):
        pairs.setdefault(game.opening_index, []).append(score)
    pair_scores = np.array([np.mean(v) for v in pairs.values()])
    if len(pair_scores) > 1:
        se = float(pair_scores.std(ddof=1) / math.sqrt(len(pair_scores)))
        low, high = mean - 1.96 * se, mean + 1.96 * se
    else:
        se, low, high = float("nan"), float("nan"), float("nan")
    terminations: dict[str, int] = {}
    for game in games:
        terminations[game.termination] = terminations.get(game.termination, 0) + 1

    def side(color: str) -> dict:
        own = [g for g in games if getattr(g, color) == name_a]
        return {"games": len(own),
                "score": float(np.mean([g.score_for(name_a) for g in own])) if own else None}

    return {
        "player_a": name_a, "player_b": name_b, "games": n, "openings": len(pairs),
        "a_wins": wins, "draws": draws, "a_losses": losses,
        "a_score": mean, "a_score_ci95": [low, high],
        "elo_diff": _elo(mean) if n else None,
        "elo_diff_ci95": [_elo(low), _elo(high)] if not math.isnan(low) else None,
        "a_as_white": side("white"), "a_as_black": side("black"),
        "terminations": dict(sorted(terminations.items())),
        "mean_plies": float(np.mean([g.plies for g in games])) if n else None,
        "identical_pair_games": sum(
            1 for i in range(0, n - 1, 2) if games[i].moves == games[i + 1].moves
        ),
    }


def to_pgn(game: GameRecord, event: str = "IT3160 arena", round_number: int | None = None) -> str:
    pgn = chess.pgn.Game()
    pgn.setup(chess.Board(game.opening_fen))
    pgn.headers.update({
        "Event": event, "Site": "local", "Round": str(round_number or game.opening_index + 1),
        "White": game.white, "Black": game.black, "Result": game.result,
        "Termination": game.termination, "PlyCount": str(game.plies),
    })
    node = pgn
    for uci in game.moves:
        node = node.add_variation(chess.Move.from_uci(uci))
    return str(pgn)


def openings_from_meta(
    fens: np.ndarray,
    plies: np.ndarray,
    count: int,
    *,
    min_ply: int = 8,
    max_ply: int = 12,
    seed: int = 0,
) -> list[str]:
    """Distinct positions (by piece placement + side + castling + ep) in a ply window."""
    candidates = sorted({
        " ".join(fen.split()[:4]) + " 0 1"
        for fen, ply in zip(fens, plies) if min_ply <= int(ply) <= max_ply
    })
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} distinct openings in plies {min_ply}-{max_ply}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[i] for i in sorted(chosen)]


def game_to_dict(game: GameRecord) -> dict:
    return asdict(game)
