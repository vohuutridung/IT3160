
"""
Classifies positions into:
- Opening
- Middlegame
- Endgame

The classification follows the fixed rules defined in the project proposal.
"""

import chess
from .schema import Phase


OPENING_MAX_PLY = 24
ENDGAME_MATERIAL_MAX = 8

MATERIAL_WEIGHTS = {
    chess.QUEEN: 4,
    chess.ROOK: 2,
    chess.BISHOP: 1,
    chess.KNIGHT: 1,
}


def material_score(board: chess.Board) -> int:
    """
    Compute the material-based game-phase score.
    G(s) = 4*N_Q + 2*N_R + N_B + N_N
    Piece counts include both White and Black.
    """

    if not isinstance(board, chess.Board):
        raise TypeError("board must be a chess.Board")

    score = 0

    for piece_type, weight in MATERIAL_WEIGHTS.items():
        num_white = len(board.pieces(piece_type, chess.WHITE))
        num_black = len(board.pieces(piece_type, chess.BLACK))

        score += weight * (num_white + num_black)

    return score


def classify_phase(
    board: chess.Board,
    ply: int | None = None,
    *,
    opening_max_ply: int = OPENING_MAX_PLY,
    endgame_material_max: int = ENDGAME_MATERIAL_MAX,
) -> Phase:
    """
    Classify a chess position into a game phase.

    Rules (applied in order):
        1. ply <= opening_max_ply -> Opening
        2. material_score <= endgame_material_max -> Endgame
        3. Otherwise -> Middlegame

    Args:
        board:
            The chess position BEFORE a move.
        ply:
            Zero-based number of half-moves already played.
            Defaults to board.ply().
        opening_max_ply:
            Opening boundary, default 24.
        endgame_material_max:
            Endgame material threshold, default 8.

    Returns:
        A Phase enum value.
    """

    if not isinstance(board, chess.Board):
        raise TypeError("board must be a chess.Board")

    for name, value in (
        ("opening_max_ply", opening_max_ply),
        ("endgame_material_max", endgame_material_max),
        ("ply", ply),
    ):
        if name == "ply" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    board_ply = board.ply()

    if ply is None:
        ply = board_ply
    elif ply != board_ply:
        raise ValueError(
            f"Ply mismatch: supplied {ply}, "
            f"but board has ply={board_ply}"
        )

    if ply <= opening_max_ply:
        return Phase.OPENING

    if material_score(board) <= endgame_material_max:
        return Phase.ENDGAME

    return Phase.MIDDLEGAME
