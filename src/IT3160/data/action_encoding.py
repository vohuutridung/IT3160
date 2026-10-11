
"""
AlphaZero-style chess action encoding.

Action space:
    64 origin squares x 73 move planes = 4672 actions.

Planes:
    0..55  : Queen-like moves (8 directions x 7 distances)
    56..63 : Knight moves (8 offsets)
    64..72 : Underpromotions (3 directions x 3 pieces)

Flattening convention:
    action_id = plane * 64 + canonical_from_square

Compatible with:
    PyTorch policy head output (73, 8, 8) flattened in channel-first order.
"""

from numbers import Integral
import chess
from .canonical import (
    canonicalize_move,
    decanonicalize_move,
)
from .schema import ACTION_SPACE_SIZE


NUM_SQUARES = 64
NUM_MOVE_PLANES = 73

QUEEN_PLANES = 56
KNIGHT_PLANES = 8
UNDERPROMOTION_START = 64


# Each direction is (delta_file, delta_rank).
QUEEN_DIRECTIONS = (
    (0, 1),     # N
    (1, 1),     # NE
    (1, 0),     # E
    (1, -1),    # SE
    (0, -1),    # S
    (-1, -1),   # SW
    (-1, 0),    # W
    (-1, 1),    # NW
)

KNIGHT_DIRECTIONS = (
    (-2, 1),
    (-1, 2),
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
)

UNDERPROMOTION_DIRECTIONS = (
    (-1, 1),    # Forward-left
    (0, 1),     # Forward
    (1, 1),     # Forward-right
)

UNDERPROMOTION_PIECES = (
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
)


QUEEN_DIRECTION_TO_INDEX = {
    direction: index
    for index, direction in enumerate(QUEEN_DIRECTIONS)
}

KNIGHT_DIRECTION_TO_INDEX = {
    direction: index
    for index, direction in enumerate(KNIGHT_DIRECTIONS)
}

UNDERPROMOTION_DIRECTION_TO_INDEX = {
    direction: index
    for index, direction in enumerate(
        UNDERPROMOTION_DIRECTIONS
    )
}

UNDERPROMOTION_PIECE_TO_INDEX = {
    piece: index
    for index, piece in enumerate(
        UNDERPROMOTION_PIECES
    )
}


def _validate_board(board: chess.Board) -> None:
    """Ensure a supported standard-chess board."""

    if not isinstance(board, chess.Board):
        raise TypeError("board must be a chess.Board")

    if board.chess960:
        raise ValueError(
            "Chess960 is not supported by this action encoder"
        )


def _sign(value: int) -> int:
    """Return -1, 0, or 1."""

    return (value > 0) - (value < 0)


def _encode_legal_move(
    board: chess.Board,
    move: chess.Move,
) -> int:
    """
    Encode a move already known to be legal.
    Internal function used by encode_move() and legal_action_ids().
    Precondition: move is legal on the original board.
    """

    canonical = canonicalize_move(
        move,
        board.turn,
    )

    from_square = canonical.from_square
    to_square = canonical.to_square

    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)

    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)

    df = to_file - from_file
    dr = to_rank - from_rank

    # Case 1: Underpromotion.
    if move.promotion in UNDERPROMOTION_PIECE_TO_INDEX:
        direction = (df, dr)

        if (
            direction not in
            UNDERPROMOTION_DIRECTION_TO_INDEX
            or from_rank != 6
            or to_rank != 7
        ):
            raise ValueError(
                f"Invalid underpromotion geometry: {move.uci()}"
            )

        direction_index = (
            UNDERPROMOTION_DIRECTION_TO_INDEX[direction]
        )
        piece_index = (
            UNDERPROMOTION_PIECE_TO_INDEX[move.promotion]
        )

        plane = (
            UNDERPROMOTION_START
            + direction_index * 3
            + piece_index
        )

    # Case 2: Knight-like geometry.
    elif (df, dr) in KNIGHT_DIRECTION_TO_INDEX:
        plane = (
            QUEEN_PLANES
            + KNIGHT_DIRECTION_TO_INDEX[(df, dr)]
        )

    # Case 3: Queen-like geometry.
    else:
        distance = max(abs(df), abs(dr))

        if distance == 0 or distance > 7:
            raise ValueError(
                f"Invalid move distance: {move.uci()}"
            )

        if not (
            df == 0
            or dr == 0
            or abs(df) == abs(dr)
        ):
            raise ValueError(
                f"Unsupported move geometry: {move.uci()}"
            )

        direction = (
            _sign(df),
            _sign(dr),
        )

        direction_index = (
            QUEEN_DIRECTION_TO_INDEX[direction]
        )

        plane = direction_index * 7 + (distance - 1)

    action_id = plane * NUM_SQUARES + from_square

    if not 0 <= action_id < ACTION_SPACE_SIZE:
        raise AssertionError("Encoded action ID out of range")

    return action_id


def encode_move(
    board: chess.Board,
    move: chess.Move,
) -> int:
    """
    Convert an ORIGINAL legal chess.Move to an action ID.
    Args:
        board: Original board before making the move.
        move: Legal move in original coordinates.
    Returns:
        Integer in [0, 4671].
    Raises:
        ValueError if move is not legal.
    """

    _validate_board(board)

    if not isinstance(move, chess.Move):
        raise TypeError("move must be a chess.Move")

    if not board.is_legal(move):
        raise ValueError(
            f"Illegal move for this position: {move.uci()}"
        )

    return _encode_legal_move(board, move)


def _decode_plane(
    canonical_from: int,
    plane: int,
) -> chess.Move:
    """
    Decode a move plane to a CANONICAL chess.Move.
    The decoded move has canonical coordinates.
    Legality is checked later on the original board.
    """

    from_file = chess.square_file(canonical_from)
    from_rank = chess.square_rank(canonical_from)

    promotion = None

    # Case 1: Queen-like move.
    if plane < QUEEN_PLANES:
        direction_index = plane // 7
        distance = plane % 7 + 1

        df, dr = QUEEN_DIRECTIONS[direction_index]

        df *= distance
        dr *= distance

    # Case 2: Knight move.
    elif plane < UNDERPROMOTION_START:
        knight_index = plane - QUEEN_PLANES

        df, dr = KNIGHT_DIRECTIONS[knight_index]

    # Case 3: Underpromotion.
    else:
        offset = plane - UNDERPROMOTION_START

        direction_index = offset // 3
        piece_index = offset % 3

        df, dr = UNDERPROMOTION_DIRECTIONS[
            direction_index
        ]

        promotion = UNDERPROMOTION_PIECES[piece_index]

    to_file = from_file + df
    to_rank = from_rank + dr

    if not (
        0 <= to_file < 8
        and 0 <= to_rank < 8
    ):
        raise ValueError(
            "Action points outside the chessboard"
        )

    canonical_to = chess.square(
        to_file,
        to_rank,
    )

    return chess.Move(
        from_square=canonical_from,
        to_square=canonical_to,
        promotion=promotion,
    )


def decode_move(
    board: chess.Board,
    action_id: int,
) -> chess.Move:
    """
    Decode an action ID to an ORIGINAL legal chess.Move.

    The board is needed to:
    - Invert canonical coordinates.
    - Detect queen promotions.
    - Validate the decoded move.

    Raises:
        ValueError for an invalid or illegal action.
    """

    _validate_board(board)

    if (
        isinstance(action_id, bool)
        or not isinstance(action_id, Integral)
    ):
        raise TypeError("action_id must be an integer")

    action_id = int(action_id)

    if not 0 <= action_id < ACTION_SPACE_SIZE:
        raise ValueError(
            "action_id must be in [0, 4671]"
        )

    plane = action_id // NUM_SQUARES
    canonical_from = action_id % NUM_SQUARES

    canonical_move = _decode_plane(
        canonical_from,
        plane,
    )

    original_move = decanonicalize_move(
        canonical_move,
        board.turn,
    )

    # Queen promotions use normal queen-like planes,
    # not underpromotion planes.
    if plane < QUEEN_PLANES:
        piece = board.piece_at(
            original_move.from_square
        )

        if (
            piece is not None
            and piece.color == board.turn
            and piece.piece_type == chess.PAWN
            and chess.square_rank(
                canonical_move.to_square
            ) == 7
        ):
            original_move = chess.Move(
                from_square=original_move.from_square,
                to_square=original_move.to_square,
                promotion=chess.QUEEN,
            )

    if not board.is_legal(original_move):
        raise ValueError(
            f"Action {action_id} is illegal "
            f"for the given position"
        )

    return original_move


def legal_action_ids(
    board: chess.Board,
) -> list[int]:
    """
    Return all legal action IDs for the current position.
    """

    _validate_board(board)

    action_ids = [
        _encode_legal_move(board, move)
        for move in board.legal_moves
    ]

    if len(action_ids) != len(set(action_ids)):
        raise ValueError(
            "Action encoding collision among legal moves"
        )

    return sorted(action_ids)
