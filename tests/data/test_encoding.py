"""Independent layout examples and legal-move invariants for chess encoding."""

import random

import chess
import numpy as np
import pytest

from IT3160.data.action_encoding import decode_move, encode_move, legal_action_ids
from IT3160.data.canonical import (
    canonicalize_move,
    canonicalize_square,
    decanonicalize_move,
    encode_board,
)


def board_state(board):
    return board.fen(en_passant="fen"), tuple(board.move_stack), board.promoted


@pytest.mark.parametrize("square", range(64))
def test_square_rotation_is_identity_for_white_and_involution_for_black(square):
    assert canonicalize_square(square, chess.WHITE) == square
    assert canonicalize_square(square, chess.BLACK) == 63 - square
    assert canonicalize_square(canonicalize_square(square, chess.BLACK), chess.BLACK) == square


@pytest.mark.parametrize("square", [True, False, 1.0, "e4", None])
def test_square_requires_integer(square):
    with pytest.raises(TypeError):
        canonicalize_square(square, chess.WHITE)


@pytest.mark.parametrize("square", [-1, 64])
def test_square_bounds(square):
    with pytest.raises(ValueError):
        canonicalize_square(square, chess.WHITE)


@pytest.mark.parametrize("turn", [0, 1, None, "black"])
def test_square_requires_chess_color(turn):
    with pytest.raises(TypeError):
        canonicalize_square(chess.E4, turn)


def test_move_rotation_preserves_promotion_and_does_not_mutate_move():
    move = chess.Move.from_uci("b2a1n")
    assert canonicalize_move(move, chess.BLACK) == chess.Move.from_uci("g7h8n")
    assert decanonicalize_move(canonicalize_move(move, chess.BLACK), chess.BLACK) == move
    assert canonicalize_move(move, chess.WHITE) == move
    assert move.uci() == "b2a1n"


@pytest.mark.parametrize("move", [chess.Move.null(), chess.Move.from_uci("N@f3")])
def test_null_and_drop_moves_are_not_supported(move):
    with pytest.raises(ValueError):
        canonicalize_move(move, chess.WHITE)


def test_initial_tensor_matches_explicit_channel_first_bottom_up_layout():
    board = chess.Board()
    before = board_state(board)
    expected = np.zeros((17, 8, 8), dtype=np.uint8)
    expected[0, 1, :] = 1
    expected[6, 6, :] = 1
    for offset, rank in ((0, 0), (6, 7)):
        expected[offset + 1, rank, [1, 6]] = 1
        expected[offset + 2, rank, [2, 5]] = 1
        expected[offset + 3, rank, [0, 7]] = 1
        expected[offset + 4, rank, 3] = 1
        expected[offset + 5, rank, 4] = 1
    expected[12:16] = 1
    tensor = encode_board(board)
    np.testing.assert_array_equal(tensor, expected)
    assert tensor.dtype == np.uint8
    assert board_state(board) == before


def test_black_tensor_rotates_both_axes_and_swaps_player_piece_groups():
    board = chess.Board()
    board.push_uci("e2e4")
    black_before = board_state(board)
    black_tensor = encode_board(board)
    white_view = board.copy(stack=False)
    white_view.turn = chess.WHITE
    white_tensor = encode_board(white_view)
    np.testing.assert_array_equal(black_tensor[:6], white_tensor[6:12, ::-1, ::-1])
    np.testing.assert_array_equal(black_tensor[6:12], white_tensor[:6, ::-1, ::-1])
    assert black_tensor[5, 0, 3] == 1  # Black king e8 -> d1.
    assert black_tensor[11, 7, 3] == 1  # White king e1 -> d8.
    assert black_tensor[6, 4, 3] == 1  # White pawn e4 -> d5.
    # Preserve the raw double-pawn target even when there is no legal EP capture.
    assert black_tensor[16, 5, 3] == 1  # e3 -> d6.
    assert black_tensor[16].sum() == 1
    assert board_state(board) == black_before


@pytest.mark.parametrize(
    "turn, expected",
    [("w", (1, 0, 0, 1)), ("b", (1, 0, 0, 1))],
)
def test_asymmetric_castling_rights_use_canonical_geometric_sides(turn, expected):
    # White retains its right rook; Black retains its left rook.
    board = chess.Board(f"r3k2r/8/8/8/8/8/8/R3K2R {turn} Kq - 0 1")
    tensor = encode_board(board)
    for channel, value in enumerate(expected, start=12):
        np.testing.assert_array_equal(tensor[channel], np.full((8, 8), value, dtype=np.uint8))


def test_piece_absence_removes_castling_right_and_no_ep_is_empty():
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w KQkq - 0 1")
    tensor = encode_board(board)
    assert tensor[12:17].sum() == 0


def test_castling_planes_reflect_rook_and_king_moves_from_original_board():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    board.push_uci("h1h2")
    # It is now Black's turn: White's remaining left rook becomes spatially right.
    assert tuple(encode_board(board)[12:16, 0, 0]) == (1, 1, 1, 0)
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    board.push_uci("e1e2")
    assert tuple(encode_board(board)[12:16, 0, 0]) == (1, 1, 0, 0)


@pytest.mark.parametrize(
    "fen, uci, expected_id",
    [
        (chess.STARTING_FEN, "e2e4", 76),
        (chess.STARTING_FEN, "g1f3", 3654),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "e7e5", 75),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "g8f6", 3713),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", 964),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1", 2756),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8", 2755),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", 963),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6", 3172),
        ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 2", "e4d3", 483),
    ],
)
def test_known_action_ids_match_channel_first_policy_flattening(fen, uci, expected_id):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert board.is_legal(move)
    assert encode_move(board, move) == expected_id
    assert decode_move(board, expected_id) == move


@pytest.mark.parametrize(
    "uci, expected_id",
    [("d4d5", 27), ("d4e5", 475), ("d4e4", 923), ("d4e3", 1371),
     ("d4d3", 1819), ("d4c3", 2267), ("d4c4", 2715), ("d4c5", 3163)],
)
def test_all_eight_queen_direction_planes_have_known_ids(uci, expected_id):
    board = chess.Board("k7/8/8/8/3Q4/8/8/7K w - - 0 1")
    assert encode_move(board, chess.Move.from_uci(uci)) == expected_id
    assert decode_move(board, expected_id).uci() == uci


@pytest.mark.parametrize(
    "uci, expected_id",
    [("d4b5", 3611), ("d4c6", 3675), ("d4e6", 3739), ("d4f5", 3803),
     ("d4f3", 3867), ("d4e2", 3931), ("d4c2", 3995), ("d4b3", 4059)],
)
def test_all_eight_knight_direction_planes_have_known_ids(uci, expected_id):
    board = chess.Board("k7/8/8/8/3N4/8/8/7K w - - 0 1")
    assert encode_move(board, chess.Move.from_uci(uci)) == expected_id
    assert decode_move(board, expected_id).uci() == uci


@pytest.mark.parametrize("distance", range(1, 8))
def test_every_queen_plane_distance_matches_expected_flattening(distance):
    board = chess.Board("8/7k/8/8/8/8/8/QK6 w - - 0 1")
    move = chess.Move(chess.A1, chess.square(0, distance))
    expected_id = (distance - 1) * 64
    assert encode_move(board, move) == expected_id
    assert decode_move(board, expected_id) == move


@pytest.mark.parametrize(
    "fen, targets",
    [
        (
            "r1r4k/1P6/8/8/8/8/8/7K w - - 0 1",
            {
                "b7a8q": 3185, "b7b8q": 49, "b7c8q": 497,
                "b7a8n": 4145, "b7a8b": 4209, "b7a8r": 4273,
                "b7b8n": 4337, "b7b8b": 4401, "b7b8r": 4465,
                "b7c8n": 4529, "b7c8b": 4593, "b7c8r": 4657,
            },
        ),
        (
            "7k/8/8/8/8/8/1p6/R1R4K b - - 0 1",
            {
                "b2a1q": 502, "b2b1q": 54, "b2c1q": 3190,
                "b2a1n": 4534, "b2a1b": 4598, "b2a1r": 4662,
                "b2b1n": 4342, "b2b1b": 4406, "b2b1r": 4470,
                "b2c1n": 4150, "b2c1b": 4214, "b2c1r": 4278,
            },
        ),
    ],
)
def test_every_promotion_piece_and_direction_for_both_colors(fen, targets):
    board = chess.Board(fen)
    actual = {move.uci(): encode_move(board, move) for move in board.legal_moves if move.promotion}
    assert actual == targets
    assert len(set(targets.values())) == 12
    for uci, action_id in targets.items():
        assert decode_move(board, action_id) == chess.Move.from_uci(uci)


@pytest.mark.parametrize("seed", [0, 11, 42])
def test_injective_roundtrip_across_deterministic_legal_playouts(seed):
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(160):
        before = board_state(board)
        moves = sorted(board.legal_moves, key=chess.Move.uci)
        actions = [encode_move(board, move) for move in moves]
        assert len(actions) == len(set(actions))
        assert all(0 <= action < 4672 for action in actions)
        assert legal_action_ids(board) == sorted(actions)
        assert [decode_move(board, action) for action in actions] == moves
        assert board_state(board) == before
        if not moves:
            break
        board.push(rng.choice(moves))


def test_initial_board_accepts_exactly_twenty_actions_in_full_action_space():
    board = chess.Board()
    valid_ids = set(legal_action_ids(board))
    assert len(valid_ids) == 20
    for action_id in range(4672):
        if action_id in valid_ids:
            assert board.is_legal(decode_move(board, action_id))
        else:
            with pytest.raises(ValueError):
                decode_move(board, action_id)


@pytest.mark.parametrize("action_id", [True, False, 76.0, "76", None])
def test_decode_requires_integer_action_id(action_id):
    with pytest.raises(TypeError):
        decode_move(chess.Board(), action_id)


def test_decode_accepts_numpy_integer_loaded_from_dataset():
    assert decode_move(chess.Board(), np.int64(76)).uci() == "e2e4"


@pytest.mark.parametrize("action_id", [-1, 4672, 2**80])
def test_decode_rejects_out_of_range_action_ids(action_id):
    with pytest.raises(ValueError):
        decode_move(chess.Board(), action_id)


@pytest.mark.parametrize("uci", ["e2e5", "a1a8", "0000"])
def test_encode_rejects_illegal_moves(uci):
    with pytest.raises(ValueError):
        encode_move(chess.Board(), chess.Move.from_uci(uci))


@pytest.mark.parametrize("fen", ["7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"])
def test_terminal_boards_have_no_legal_actions(fen):
    assert legal_action_ids(chess.Board(fen)) == []


def test_encoding_rejects_unsupported_board_or_move_types():
    with pytest.raises(TypeError):
        encode_board("fen")
    with pytest.raises(TypeError):
        canonicalize_move("e2e4", chess.WHITE)
    with pytest.raises(TypeError):
        encode_move(chess.Board(), "e2e4")
    for operation in (
        lambda board: encode_move(board, chess.Move.from_uci("e2e4")),
        lambda board: decode_move(board, 76),
        legal_action_ids,
    ):
        with pytest.raises(TypeError):
            operation(None)
        with pytest.raises(ValueError, match="Chess960"):
            operation(chess.Board(chess960=True))
