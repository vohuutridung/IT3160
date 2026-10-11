
"""
Canonical chess board representation.
- Convert squares to side-to-move canonical coordinates.
- Convert moves to/from canonical coordinates.
- Encode a chess.Board into a 17 x 8 x 8 tensor.

Convention:
- White to move: identity transformation.
- Black to move: rotate all square coordinates by 180 degrees.
- Channels 0..5: current player's pieces.
- Channels 6..11: opponent's pieces.
- Castling sides follow canonical geometry.
- Row index 0 corresponds to rank 1 (bottom-up indexing).

Important:
The original chess.Board is never modified.
All legal move generation must use the original board.
"""

import chess
import numpy as np
from .schema import BOARD_SHAPE


PIECE_CHANNELS = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

CASTLING_CHANNEL_START = 12
EN_PASSANT_CHANNEL = 16


def canonicalize_square(
    square: int,
    turn: bool,
) -> int:
    """
    Convert a square to side-to-move canonical coordinates.
    White to move: identity.
    Black to move: rotate 180 degrees, i.e. square -> 63 - square.
    """

    if isinstance(square, bool) or not isinstance(square, int):
        raise TypeError("square must be an integer")

    if not 0 <= square < 64:
        raise ValueError("square must be in [0, 63]")

    if not isinstance(turn, bool):
        raise TypeError("turn must be chess.WHITE or chess.BLACK")

    if turn == chess.WHITE:
        return square

    return 63 - square


def canonicalize_move(
    move: chess.Move,
    turn: bool,
) -> chess.Move:
    """
    Convert a move to canonical coordinates.
    Transform: From-square, To-square. Preserve: Promotion piece type.
    """

    if not isinstance(move, chess.Move):
        raise TypeError("move must be a chess.Move")

    if not move:
        raise ValueError("Null moves are not supported")

    if move.drop is not None:
        raise ValueError("Drop moves are not supported")

    from_square = canonicalize_square(
        move.from_square,
        turn,
    )

    to_square = canonicalize_square(
        move.to_square,
        turn,
    )

    return chess.Move(
        from_square=from_square,
        to_square=to_square,
        promotion=move.promotion,
    )


def decanonicalize_move(
    move: chess.Move,
    turn: bool,
) -> chess.Move:
    """
    Convert a canonical move back to original coordinates.
    """

    return canonicalize_move(move, turn)


def _canonical_castling_rights(
    board: chess.Board,
) -> tuple[bool, bool, bool, bool]:
    """
    Return castling rights in canonical spatial coordinates.
    Order: my_kingside, my_queenside, opponent_kingside, opponent_queenside
    For Black to move, 180-degree rotation swaps the geometric kingside/queenside directions.
    """

    me = board.turn
    opponent = not me

    if me == chess.WHITE:
        return (
            board.has_kingside_castling_rights(me),
            board.has_queenside_castling_rights(me),
            board.has_kingside_castling_rights(opponent),
            board.has_queenside_castling_rights(opponent),
        )

    return (
        board.has_queenside_castling_rights(me),
        board.has_kingside_castling_rights(me),
        board.has_queenside_castling_rights(opponent),
        board.has_kingside_castling_rights(opponent),
    )


def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a chess position into a canonical 17x8x8 tensor.
    Returns: np.ndarray of shape (17, 8, 8), dtype uint8.
    Spatial layout: tensor[:, 0, :] corresponds to rank 1, tensor[:, 7, :] corresponds to rank 8
    Channel layout: 0..5: my pieces (P, N, B, R, Q, K), 6..11: opponent pieces (P, N, B, R, Q, K), 12..15: canonical castling rights, 16: en-passant target square
    """

    if not isinstance(board, chess.Board):
        raise TypeError("board must be a chess.Board")

    tensor = np.zeros(
        BOARD_SHAPE,
        dtype=np.uint8,
    )

    turn = board.turn

    # 1. Encode piece placement.
    for square, piece in board.piece_map().items():
        canonical_square = canonicalize_square(
            square,
            turn,
        )

        rank = chess.square_rank(canonical_square)
        file = chess.square_file(canonical_square)

        channel = PIECE_CHANNELS[piece.piece_type]

        if piece.color != turn:
            channel += 6

        tensor[channel, rank, file] = 1

    # 2. Encode castling rights.
    castling_rights = _canonical_castling_rights(board)

    for offset, has_right in enumerate(castling_rights):
        channel = CASTLING_CHANNEL_START + offset

        tensor[channel, :, :] = int(has_right)

    # 3. Encode en-passant target square.
    if board.ep_square is not None:
        canonical_ep = canonicalize_square(
            board.ep_square,
            turn,
        )

        rank = chess.square_rank(canonical_ep)
        file = chess.square_file(canonical_ep)

        tensor[EN_PASSANT_CHANNEL, rank, file] = 1

    return tensor
