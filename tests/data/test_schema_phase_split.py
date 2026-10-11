"""Record contracts, proposal phase boundaries, and stable game splits."""

from dataclasses import FrozenInstanceError
import json
import math

import chess
import pytest

from IT3160.data.phase import classify_phase, material_score
from IT3160.data.schema import CandidateRecord, Phase, ProcessedRecord, Split
from IT3160.data.split import SplitRatios, _hash_fraction, assign_split


def candidate(**changes):
    fields = dict(
        game_id="game-0", source_month="2026-09", split=Split.TRAIN,
        phase=Phase.OPENING, ply=0, fen=chess.STARTING_FEN, move_uci="e2e4",
        white_elo=2500, black_elo=2600, result="1-0",
    )
    fields.update(changes)
    return CandidateRecord(**fields)


def processed(**changes):
    fields = candidate().to_dict()
    fields.update(split=Split.TRAIN, phase=Phase.OPENING, position_hash="hash",
                  action_id=76, legal_action_ids=(12, 76, 3654))
    fields.update(changes)
    return ProcessedRecord(**fields)


def test_records_serialize_plain_values_and_are_immutable():
    record = candidate()
    serialized = record.to_dict()
    assert serialized["phase"] == "opening"
    assert serialized["split"] == "train"
    assert json.loads(json.dumps(serialized)) == serialized
    serialized["game_id"] = "changed"
    assert record.game_id == "game-0"
    with pytest.raises(FrozenInstanceError):
        record.ply = 1
    output = processed().to_dict()
    assert output["legal_action_ids"] == [12, 76, 3654]
    assert json.loads(json.dumps(output)) == output


@pytest.mark.parametrize("factory", [candidate, processed])
@pytest.mark.parametrize("field", ["game_id", "source_month", "fen", "move_uci"])
@pytest.mark.parametrize("value, error", [(None, TypeError), (123, TypeError), ("", ValueError), ("  ", ValueError)])
def test_required_record_fields_are_nonblank_strings(factory, field, value, error):
    with pytest.raises(error):
        factory(**{field: value})


@pytest.mark.parametrize("source_month", ["2026-00", "2026-13", "26-09", "2026-9", "2026-09 ", "２０２６-09"])
def test_source_month_uses_ascii_year_month(source_month):
    with pytest.raises(ValueError):
        candidate(source_month=source_month)


@pytest.mark.parametrize("factory", [candidate, processed])
@pytest.mark.parametrize("field", ["ply", "white_elo", "black_elo"])
@pytest.mark.parametrize("value, error", [(True, TypeError), (0.0, TypeError), ("1", TypeError), (-1, ValueError)])
def test_record_counts_and_ratings_require_nonnegative_integers(factory, field, value, error):
    with pytest.raises(error):
        factory(**{field: value})


@pytest.mark.parametrize("factory", [candidate, processed])
def test_records_require_real_phase_and_split_enums(factory):
    with pytest.raises(TypeError):
        factory(phase="opening")
    with pytest.raises(TypeError):
        factory(split="train")


@pytest.mark.parametrize("result", [None, "1-0", "0-1", "1/2-1/2", "*"])
def test_optional_metadata_and_conventional_pgn_results(result):
    assert candidate(white_elo=None, black_elo=None, result=result).result == result


@pytest.mark.parametrize("result, error", [(1, TypeError), ("", ValueError), ("unknown", ValueError)])
def test_result_metadata_validation(result, error):
    with pytest.raises(error):
        candidate(result=result)


@pytest.mark.parametrize("changes, error", [
    ({"position_hash": " "}, ValueError),
    ({"position_hash": 123}, TypeError),
    ({"action_id": True}, TypeError),
    ({"action_id": 76.0}, TypeError),
    ({"action_id": -1}, ValueError),
    ({"action_id": 4672}, ValueError),
    ({"legal_action_ids": []}, TypeError),
    ({"legal_action_ids": ()}, ValueError),
    ({"legal_action_ids": (76, 76)}, ValueError),
    ({"legal_action_ids": (76, -1)}, ValueError),
    ({"legal_action_ids": (76, 4672)}, ValueError),
    ({"legal_action_ids": (76, True)}, TypeError),
    ({"legal_action_ids": (76.0,)}, TypeError),
    ({"legal_action_ids": (12,)}, ValueError),
])
def test_processed_record_action_contract(changes, error):
    with pytest.raises(error):
        processed(**changes)


def test_chess_semantics_remain_available_to_record_validator():
    # Schema checks primitives; validation.py reports invalid chess metadata.
    assert candidate(fen="bad FEN", move_uci="bad UCI").fen == "bad FEN"


def test_material_score_counts_both_colors_and_ignores_pawns_and_kings():
    assert material_score(chess.Board()) == 24
    assert material_score(chess.Board("7k/pp6/8/8/8/8/PP6/K7 w - - 0 1")) == 0
    board = chess.Board("qrbn3k/8/8/8/8/8/8/QRBN3K w - - 0 1")
    assert material_score(board) == 16


def test_opening_boundary_and_precedence_at_exact_ply():
    for turn, fullmove, expected in ((chess.WHITE, 13, Phase.OPENING), (chess.BLACK, 13, Phase.MIDDLEGAME)):
        board = chess.Board()
        board.turn = turn
        board.fullmove_number = fullmove
        assert classify_phase(board) == expected
        assert classify_phase(board, board.ply()) == expected
    kings_only = chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 13")
    assert kings_only.ply() == 24
    assert classify_phase(kings_only) == Phase.OPENING
    kings_only.turn = chess.BLACK
    assert classify_phase(kings_only) == Phase.ENDGAME


def test_endgame_includes_material_eight_but_not_nine():
    eight = chess.Board("3q3k/8/8/8/8/8/8/K2Q4 b - - 0 13")
    nine = chess.Board("3q3k/8/8/8/8/8/8/KN1Q4 b - - 0 13")
    assert material_score(eight) == 8
    assert material_score(nine) == 9
    assert classify_phase(eight) == Phase.ENDGAME
    assert classify_phase(nine) == Phase.MIDDLEGAME


def test_phase_custom_thresholds_and_ply_consistency():
    board = chess.Board()
    board.turn = chess.BLACK
    assert classify_phase(board, opening_max_ply=0) == Phase.MIDDLEGAME
    assert classify_phase(board, opening_max_ply=0, endgame_material_max=24) == Phase.ENDGAME
    with pytest.raises(ValueError, match="Ply mismatch"):
        classify_phase(board, ply=0)


@pytest.mark.parametrize("field", ["ply", "opening_max_ply", "endgame_material_max"])
@pytest.mark.parametrize("value, error", [(True, TypeError), (1.0, TypeError), ("1", TypeError), (-1, ValueError)])
def test_phase_arguments_require_nonnegative_integers(field, value, error):
    with pytest.raises(error):
        classify_phase(chess.Board(), **{field: value})


def test_phase_operations_require_board():
    with pytest.raises(TypeError):
        material_score(None)
    with pytest.raises(TypeError):
        classify_phase(None)


@pytest.mark.parametrize("ratios", [
    (0, 0.5, 0.5), (-0.1, 0.55, 0.55), (True, 0.0, 0.0),
    (math.inf, 0.05, 0.05), (math.nan, 0.05, 0.05),
    ("0.9", 0.05, 0.05), (0.8, 0.05, 0.05),
])
def test_split_ratios_reject_invalid_probabilities(ratios):
    with pytest.raises(ValueError):
        SplitRatios(*ratios)


def test_split_defaults_and_small_sum_rounding_tolerance():
    assert SplitRatios() == SplitRatios(0.9, 0.05, 0.05)
    assert SplitRatios(0.9 + 1e-10, 0.05, 0.05).train > 0.9


def test_sha256_version_one_known_split_assignments():
    assert assign_split("game-0") == Split.TRAIN
    assert assign_split("game-15") == Split.VALIDATION
    assert assign_split("game-1") == Split.TEST
    ids = [f"game-{i}" for i in range(100)]
    expected = {game_id: assign_split(game_id) for game_id in ids}
    assert {game_id: assign_split(game_id) for game_id in reversed(ids)} == expected
    assert any(assign_split(game_id, seed=43) != expected[game_id] for game_id in ids)


def test_split_thresholds_are_half_open(monkeypatch):
    import IT3160.data.split as module
    for fraction, expected in ((0.0, Split.TRAIN), (0.899999, Split.TRAIN),
                               (0.90, Split.VALIDATION), (0.949999, Split.VALIDATION),
                               (0.9500000000000001, Split.TEST), (math.nextafter(1.0, 0.0), Split.TEST)):
        monkeypatch.setattr(module, "_hash_fraction", lambda game_id, seed: fraction)
        assert assign_split("game") == expected


def test_hash_fraction_never_rounds_up_to_one(monkeypatch):
    import IT3160.data.split as module

    class MaximumDigest:
        def digest(self):
            return b"\xff" * 32

    monkeypatch.setattr(module.hashlib, "sha256", lambda payload: MaximumDigest())
    assert _hash_fraction("game", 42) == math.nextafter(1.0, 0.0)
    assert 0 <= _hash_fraction("game", 42) < 1


@pytest.mark.parametrize("game_id", ["", "  ", None, 123])
def test_split_requires_nonblank_game_id(game_id):
    with pytest.raises(ValueError):
        assign_split(game_id)


@pytest.mark.parametrize("seed", [True, 42.0, "42", None])
def test_split_requires_integer_seed(seed):
    with pytest.raises(TypeError):
        assign_split("game", seed=seed)


def test_split_requires_ratios_object():
    with pytest.raises(TypeError):
        assign_split("game", ratios=(0.9, 0.05, 0.05))
