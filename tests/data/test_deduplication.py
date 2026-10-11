"""Persistent deduplication and chess position equivalence regressions."""

import io
import sqlite3
from contextlib import closing

import chess
import chess.pgn
import numpy as np
import pytest

from IT3160.data.canonical import encode_board
from IT3160.data import deduplication
from IT3160.data.deduplication import (
    GAME_HASH_VERSION,
    POSITION_HASH_VERSION,
    DeduplicationIndex,
    game_content_hash,
    position_hash,
)
from IT3160.data.schema import Split


def read_game(text):
    game = chess.pgn.read_game(io.StringIO(text))
    assert game is not None and not game.errors
    return game


def test_complete_game_hash_ignores_headers_variations_and_annotations():
    first = read_game('[Event "A"]\n\n1. e4 e5 2. Nf3 Nc6 *')
    annotated = read_game('[Event "B"]\n\n1. e4 {note} e5 2. Nf3 (2. Bc4) Nc6 *')
    changed = read_game('1. e4 e5 2. Nf3 Nf6 *')
    assert game_content_hash(first) == game_content_hash(annotated)
    assert game_content_hash(first) != game_content_hash(changed)
    assert len(game_content_hash(first)) == 32
    assert first.end().board().fullmove_number == 3


def test_game_hash_includes_initial_position_and_all_moves():
    game = read_game('1. e4 e5 2. Nf3 Nc6 *')
    short = read_game('1. e4 e5 *')
    alternative = chess.pgn.Game()
    board = chess.Board()
    board.remove_piece_at(chess.A2)
    alternative.setup(board)
    node = alternative
    for move in game.mainline_moves():
        node = node.add_variation(move)
    assert game_content_hash(game) != game_content_hash(short)
    assert game_content_hash(game) != game_content_hash(alternative)
    with pytest.raises(TypeError):
        game_content_hash(chess.Board())


def test_position_hash_ignores_move_counters_and_preserves_board():
    board = chess.Board()
    board.push_uci("e2e4")
    before_fen = board.fen(en_passant="fen")
    before_stack = list(board.move_stack)
    different_counters = board.copy()
    different_counters.halfmove_clock = 63
    different_counters.fullmove_number = 99
    assert position_hash(board) == position_hash(different_counters)
    assert board.fen(en_passant="fen") == before_fen
    assert board.move_stack == before_stack
    with pytest.raises(TypeError):
        position_hash("not a board")


def test_position_hash_retains_castling_rights():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    no_white_kingside = board.copy()
    no_white_kingside.castling_rights &= ~chess.BB_H1
    no_rights = board.copy()
    no_rights.castling_rights = 0
    assert len({position_hash(board), position_hash(no_white_kingside), position_hash(no_rights)}) == 3


@pytest.mark.parametrize("fen", [
    # A FEN-format target exists after a double pawn push, without a capturer.
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    # An adjacent pawn cannot capture because it would expose its king.
    "k7/8/8/4KPpr/8/8/8/8 w - g6 0 1",
])
def test_nonlegal_en_passant_is_normalized_only_for_dedup_hash(fen):
    board = chess.Board(fen)
    assert board.is_valid()
    assert board.ep_square is not None
    assert not board.has_legal_en_passant()
    without_ep = board.copy()
    without_ep.ep_square = None
    assert not np.array_equal(encode_board(board), encode_board(without_ep))
    assert position_hash(board) == position_hash(without_ep)
    assert board.ep_square is not None


def test_legal_en_passant_distinguishes_position_hash():
    board = chess.Board("k7/8/8/4KPp1/8/8/8/8 w - g6 0 1")
    assert board.is_valid() and board.has_legal_en_passant()
    without_ep = board.copy()
    without_ep.ep_square = None
    assert position_hash(board) != position_hash(without_ep)


def test_canonical_color_rotation_shares_hash():
    board = chess.Board("4k3/8/3n4/8/3P4/2N5/8/4K3 w - - 0 1")
    equivalent = chess.Board(None)
    for square, piece in board.piece_map().items():
        equivalent.set_piece_at(63 - square, chess.Piece(piece.piece_type, not piece.color))
    equivalent.turn = not board.turn
    assert board.is_valid() and equivalent.is_valid()
    assert np.array_equal(encode_board(board), encode_board(equivalent))
    assert position_hash(board) == position_hash(equivalent)


def test_index_removes_all_cross_split_positions_but_keeps_intrasplit_repeats(tmp_path):
    path = tmp_path / "nested" / "dedup.sqlite"
    with DeduplicationIndex(path) as index:
        assert index.register_game("same game")
        assert not index.register_game("same game")
        assert index.register_game("other game")
        index.register_positions(iter([
            ("train-only", Split.TRAIN), ("train-only", Split.TRAIN),
            ("validation-only", Split.VALIDATION), ("test-only", Split.TEST),
            ("two-way", Split.TRAIN), ("two-way", Split.TEST),
            ("three-way", Split.TRAIN), ("three-way", Split.VALIDATION), ("three-way", Split.TEST),
        ]))
        with pytest.raises(RuntimeError, match="Finalize"):
            index.has_cross_split_conflict("two-way")
        index.finalize()
        index.finalize()
        assert index.is_safe_position("train-only", Split.TRAIN)
        assert not index.is_safe_position("train-only", Split.TEST)
        assert not index.has_cross_split_conflict("train-only")
        for fingerprint in ("two-way", "three-way"):
            assert index.has_cross_split_conflict(fingerprint)
            assert all(not index.is_safe_position(fingerprint, split) for split in Split)
        assert index.stats() == {
            "unique_game_contents": 2, "unique_position_hashes": 5,
            "cross_split_position_hashes": 2, "indexed_candidate_occurrences": 9,
            "status": "finalized", "position_hash_version": POSITION_HASH_VERSION,
            "game_hash_version": GAME_HASH_VERSION,
        }
    with DeduplicationIndex(path) as reopened:
        assert reopened.is_safe_position("test-only", Split.TEST)
        assert reopened.stats()["indexed_candidate_occurrences"] == 9
        with pytest.raises(RuntimeError, match="finalized"):
            reopened.register_game("new")
        with pytest.raises(RuntimeError, match="finalized"):
            reopened.register_position("new", Split.TRAIN)
        with pytest.raises(RuntimeError, match="finalized"):
            reopened.register_positions([])
        with pytest.raises(KeyError, match="not registered"):
            reopened.has_cross_split_conflict("absent")


def test_index_is_disk_backed_and_streams_register_positions(tmp_path):
    path = tmp_path / "dedup.sqlite"
    with DeduplicationIndex(path) as index:
        index.register_positions((f"hash-{i}", Split.TRAIN) for i in range(1000))
        index.commit()
        with closing(sqlite3.connect(path)) as independent:
            assert independent.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 1000
        assert index.stats()["unique_position_hashes"] == 1000


def test_context_exception_rolls_back_pending_records(tmp_path):
    path = tmp_path / "dedup.sqlite"
    with pytest.raises(RuntimeError, match="cancel"):
        with DeduplicationIndex(path) as index:
            index.register_game("uncommitted")
            index.register_position("uncommitted", Split.TRAIN)
            raise RuntimeError("cancel")
    with DeduplicationIndex(path) as reopened:
        assert reopened.stats()["unique_game_contents"] == 0
        assert reopened.stats()["unique_position_hashes"] == 0


@pytest.mark.parametrize("key", ["position_hash_version", "game_hash_version"])
def test_index_rejects_incompatible_hash_versions(tmp_path, key):
    path = tmp_path / "dedup.sqlite"
    with DeduplicationIndex(path):
        pass
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("UPDATE metadata SET value='old-incompatible' WHERE key=?", (key,))
        conn.commit()
    with pytest.raises(ValueError, match="Incompatible index version"):
        DeduplicationIndex(path)


@pytest.mark.parametrize("key", ["position_hash_version", "game_hash_version"])
def test_constructor_closes_connection_when_version_check_fails(tmp_path, monkeypatch, key):
    path = tmp_path / "dedup.sqlite"
    with DeduplicationIndex(path):
        pass
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("UPDATE metadata SET value='incompatible' WHERE key=?", (key,))
        conn.commit()

    real_connect = sqlite3.connect
    connections = []

    class TrackedConnection:
        def __init__(self, connection):
            self.connection = connection
            self.closed = False

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def close(self):
            self.closed = True
            self.connection.close()

    def tracked_connect(*args, **kwargs):
        connection = TrackedConnection(real_connect(*args, **kwargs))
        connections.append(connection)
        return connection

    monkeypatch.setattr(deduplication.sqlite3, "connect", tracked_connect)
    with pytest.raises(ValueError, match="Incompatible index version"):
        DeduplicationIndex(path)
    assert len(connections) == 1
    assert connections[0].closed
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].connection.execute("SELECT 1")


@pytest.mark.parametrize("fingerprint", [None, "", 123])
def test_index_validates_fingerprints(tmp_path, fingerprint):
    with DeduplicationIndex(tmp_path / "dedup.sqlite") as index:
        with pytest.raises(ValueError, match="non-empty string"):
            index.register_game(fingerprint)
        with pytest.raises(ValueError, match="non-empty string"):
            index.register_position(fingerprint, Split.TRAIN)
        with pytest.raises(ValueError, match="non-empty string"):
            index.register_positions([(fingerprint, Split.TRAIN)])


def test_index_requires_split_enums(tmp_path):
    with DeduplicationIndex(tmp_path / "dedup.sqlite") as index:
        with pytest.raises(TypeError, match="Split enum"):
            index.register_position("hash", "train")
        index.register_position("hash", Split.TRAIN)
        index.finalize()
        with pytest.raises(TypeError, match="Split enum"):
            index.is_safe_position("hash", "train")

