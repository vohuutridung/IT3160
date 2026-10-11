"""Row corruption, aggregate leakage, quotas and bounded reporting tests."""

from dataclasses import replace

import chess
import pytest

from IT3160.data import validation
from IT3160.data.action_encoding import encode_move, legal_action_ids
from IT3160.data.deduplication import position_hash
from IT3160.data.phase import classify_phase
from IT3160.data.schema import CandidateRecord, Phase, ProcessedRecord, Split
from IT3160.data.split import assign_split


def candidate(game_id="game-1", split=None, board=None, move="e2e4"):
    board = chess.Board() if board is None else board
    return CandidateRecord(
        game_id=game_id, source_month="2025-01", split=assign_split(game_id) if split is None else split,
        phase=classify_phase(board), ply=board.ply(), fen=board.fen(en_passant="fen"),
        move_uci=move, white_elo=2500, black_elo=2400, result="1-0",
    )


def processed(game_id="game-1", split=None, board=None, move="e2e4"):
    row = candidate(game_id, split, board, move)
    values = row.to_dict()
    values.update(split=row.split, phase=row.phase)
    board = chess.Board(row.fen)
    return ProcessedRecord(
        **values, position_hash=position_hash(board),
        action_id=encode_move(board, chess.Move.from_uci(move)),
        legal_action_ids=tuple(legal_action_ids(board)),
    )


def codes(issues):
    return {issue.code for issue in issues}


@pytest.mark.parametrize("board,move,phase", [
    (chess.Board(), "e2e4", Phase.OPENING),
    (chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 14"), "d2d4", Phase.MIDDLEGAME),
    (chess.Board("4k3/8/8/8/8/8/P7/4K3 w - - 0 14"), "a2a4", Phase.ENDGAME),
    (chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), "e1g1", Phase.OPENING),
    (chess.Board("k7/8/8/4KPp1/8/8/8/8 w - g6 0 1"), "f5g6", Phase.OPENING),
    (chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 14"), "a7a8n", Phase.ENDGAME),
])
def test_valid_candidate_and_processed_rows_cover_phases_and_special_moves(board, move, phase):
    raw = candidate(board=board, move=move)
    ready = processed(board=board, move=move)
    assert raw.phase == phase
    assert validation.validate_candidate_record(raw) == []
    assert validation.validate_processed_record(ready) == []


@pytest.mark.parametrize("changes,expected", [
    ({"fen": "garbage"}, {"invalid_fen"}),
    ({"fen": "8/8/8/8/8/8/8/8 w - - 0 1"}, {"invalid_fen"}),
    ({"move_uci": "not a move"}, {"invalid_move_uci"}),
    ({"move_uci": "e2e5"}, {"illegal_human_move"}),
    ({"ply": 1}, {"ply_mismatch"}),
    ({"phase": Phase.ENDGAME}, {"phase_mismatch"}),
])
def test_candidate_corruption_returns_issues_instead_of_raising(changes, expected):
    row = replace(candidate(), **changes)
    issues = validation.validate_candidate_record(row)
    assert codes(issues) == expected
    for issue in issues:
        assert issue.game_id == row.game_id
        assert issue.split == row.split.value
        assert issue.ply == row.ply
        assert issue.to_dict()["code"] == issue.code


def test_ply_mismatch_does_not_hide_wrong_phase_or_abort_dataset_validation():
    row = replace(candidate(), ply=27, phase=Phase.MIDDLEGAME)
    assert codes(validation.validate_candidate_record(row)) == {"ply_mismatch", "phase_mismatch"}
    report = validation.validate_dataset([row], kind="candidate")
    assert not report.ok
    assert report.checked == 1 and report.error_count == 2


def test_split_recomputation_can_be_disabled_and_custom_phase_thresholds_work():
    row = candidate()
    wrong_split = next(split for split in Split if split != row.split)
    changed = replace(row, split=wrong_split)
    assert codes(validation.validate_candidate_record(changed)) == {"split_mismatch"}
    assert validation.validate_candidate_record(changed, check_split=False) == []
    board = chess.Board()
    board.push_uci("e2e4")
    black_row = candidate(board=board, move="e7e5")
    assert codes(validation.validate_candidate_record(black_row, opening_max_ply=0)) == {"phase_mismatch"}
    changed_phase = replace(black_row, phase=Phase.ENDGAME)
    assert validation.validate_candidate_record(changed_phase, opening_max_ply=0, endgame_material_max=24) == []


def test_processed_hash_action_and_legal_list_corruption_are_detected():
    row = processed()
    assert codes(validation.validate_processed_record(replace(row, position_hash="corrupt"))) == {"position_hash_mismatch"}
    board = chess.Board(row.fen)
    other_action = encode_move(board, chess.Move.from_uci("d2d4"))
    changed_target = replace(row, action_id=other_action)
    assert codes(validation.validate_processed_record(changed_target)) == {"action_id_mismatch", "action_roundtrip_mismatch"}
    narrowed = replace(row, legal_action_ids=(row.action_id,))
    assert codes(validation.validate_processed_record(narrowed)) == {"legal_action_ids_mismatch"}


def test_processed_validator_detects_corruption_beyond_constructor_guards():
    row = processed()
    object.__setattr__(row, "legal_action_ids", (4672,))
    issues = validation.validate_processed_record(row)
    assert codes(issues) == {"legal_action_ids_mismatch", "target_not_in_legal", "legal_action_out_of_range"}
    for issue in issues:
        assert issue.position_hash == row.position_hash


@pytest.mark.parametrize("changes,expected", [
    ({"fen": "garbage"}, {"invalid_fen"}),
    ({"move_uci": "invalid"}, {"invalid_move_uci"}),
    ({"move_uci": "e2e5"}, {"illegal_human_move"}),
])
def test_processed_checks_stop_safely_when_board_or_human_move_is_unusable(changes, expected):
    assert codes(validation.validate_processed_record(replace(processed(), **changes))) == expected


def test_row_validators_reject_other_record_kinds():
    with pytest.raises(TypeError, match="CandidateRecord"):
        validation.validate_candidate_record(processed())
    with pytest.raises(TypeError, match="ProcessedRecord"):
        validation.validate_processed_record(candidate())


@pytest.mark.parametrize("kind,factory", [("candidate", candidate), ("processed", processed)])
def test_dataset_detects_game_phase_and_position_leakage(kind, factory):
    rows = [factory("same-game", Split.TRAIN), factory("same-game", Split.TEST)]
    report = validation.validate_dataset(iter(rows), kind=kind, check_split=False)
    assert not report.ok and report.checked == 2
    assert codes(report.issues) == {"game_cross_split", "game_phase_oversample", "position_cross_split"}
    assert report.error_count == 3
    assert report.stats["split_counts"] == {"train": 1, "test": 1}
    assert report.stats["phase_counts"] == {"test/opening": 1, "train/opening": 1}


def test_candidate_quick_validation_still_checks_position_leakage():
    rows = [candidate("a", Split.TRAIN), candidate("b", Split.VALIDATION)]
    report = validation.validate_dataset(rows, kind="candidate", row_check=False)
    assert codes(report.issues) == {"position_cross_split"}
    assert report.error_count == 1


def test_corrupt_processed_hash_cannot_hide_leakage_in_full_validation():
    rows = [processed("a", Split.TRAIN), replace(processed("b", Split.TEST), position_hash="different-but-corrupt")]
    full_report = validation.validate_dataset(rows, kind="processed", check_split=False)
    assert codes(full_report.issues) == {"position_hash_mismatch", "position_cross_split"}
    quick_report = validation.validate_dataset(rows, kind="processed", row_check=False)
    assert quick_report.ok  # Aggregate-only mode explicitly trusts stored hashes.


def test_within_split_repeated_positions_are_a_warning():
    rows = [processed(f"game-{i}", Split.TRAIN) for i in range(3)]
    report = validation.validate_dataset(rows, kind="processed", check_split=False)
    assert report.ok
    assert report.error_count == 0 and report.warning_count == 1
    assert codes(report.issues) == {"within_split_position_duplicates"}
    assert report.stats["within_split_position_duplicates"] == 2


def test_quota_counts_missing_short_and_surplus_buckets():
    quotas = {
        (Split.TRAIN, Phase.OPENING): 0,
        (Split.VALIDATION, Phase.MIDDLEGAME): 2,
        (Split.TEST, Phase.ENDGAME): 0,
    }
    report = validation.validate_dataset([candidate(split=Split.TRAIN)], kind="candidate", quotas=quotas, check_split=False)
    assert report.error_count == 1 and report.warning_count == 1
    assert codes(report.issues) == {"quota_shortfall", "quota_surplus"}
    assert report.stats["quota_shortfalls"] == {"validation/middlegame": 2}
    assert report.stats["quota"] == {
        "train/opening": {"expected": 0, "actual": 1},
        "validation/middlegame": {"expected": 2, "actual": 0},
        "test/endgame": {"expected": 0, "actual": 0},
    }


@pytest.mark.parametrize("limit", [0, 3, 30])
def test_issue_cap_preserves_full_error_counts(limit):
    rows = (replace(candidate(f"game-{i}"), fen="invalid") for i in range(20))
    report = validation.validate_dataset(rows, kind="candidate", check_split=False, max_issues=limit)
    assert report.checked == report.error_count == 20
    assert len(report.issues) == min(20, limit)
    payload = report.to_dict()
    assert payload["ok"] is False
    assert payload["issues_truncated"] == (limit < 20)


@pytest.mark.parametrize("kind,right,wrong", [("candidate", candidate, processed), ("processed", processed, candidate)])
def test_dataset_type_errors_are_counted_without_stopping(kind, right, wrong):
    report = validation.validate_dataset([right(), wrong(), None, {}], kind=kind, row_check=False)
    assert report.checked == 4
    assert report.error_count == 3
    assert codes(report.issues) == {"wrong_record_type"}


@pytest.mark.parametrize("kwargs,error", [
    ({"kind": "other"}, ValueError),
    ({"max_issues": -1}, ValueError), ({"max_issues": True}, TypeError), ({"max_issues": 1.2}, TypeError),
    ({"quotas": {(Split.TRAIN, Phase.OPENING): -1}}, ValueError),
    ({"quotas": {(Split.TRAIN, Phase.OPENING): True}}, TypeError),
    ({"quotas": {(Split.TRAIN, Phase.OPENING): 1.5}}, TypeError),
    ({"quotas": {("train", Phase.OPENING): 1}}, TypeError),
    ({"quotas": {Split.TRAIN: 1}}, TypeError),
])
def test_invalid_validation_options_fail_early(kwargs, error):
    arguments = {"kind": "candidate", **kwargs}
    with pytest.raises(error):
        validation.validate_dataset([], **arguments)


def test_dataset_index_streams_aggregate_queries_and_uses_disk(tmp_path):
    index = validation._DatasetIndex()
    try:
        assert index.path.exists() and index.path.suffix == ".sqlite"
        for i in range(100):
            for split in ("train", "test"):
                index.observe(game_id=f"game-{i}", split=split, phase="opening", position_hash_value=f"position-{i}")
        index.commit()
        for query in (index.cross_split_games, index.cross_split_positions, index.multi_phase_samples):
            rows = query()
            assert iter(rows) is rows
            assert len(list(rows)) == 100
    finally:
        index.close()
    assert not index.path.exists()


@pytest.mark.parametrize("fails", [False, True])
def test_temporary_index_is_removed_after_success_and_source_failure(monkeypatch, fails):
    paths = []
    original_init = validation._DatasetIndex.__init__

    def tracked_init(self):
        original_init(self)
        paths.append(self.path)

    monkeypatch.setattr(validation._DatasetIndex, "__init__", tracked_init)

    def source():
        assert paths[0].exists()
        yield candidate()
        if fails:
            raise RuntimeError("source failed")

    if fails:
        with pytest.raises(RuntimeError, match="source failed"):
            validation.validate_dataset(source(), kind="candidate")
    else:
        assert validation.validate_dataset(source(), kind="candidate").ok
    assert len(paths) == 1 and not paths[0].exists()


def test_small_pilot_materialization_and_report_roundtrip():
    row = candidate()
    rows, report = validation.iter_and_validate(iter([row]), kind="candidate")
    assert rows == [row]
    assert report.to_dict() == {
        "kind": "candidate", "ok": True, "checked": 1, "error_count": 0,
        "warning_count": 0, "issues": [], "max_issues": 100, "issues_truncated": False,
        "stats": {
            "split_counts": {row.split.value: 1},
            "phase_counts": {f"{row.split.value}/opening": 1},
            "within_split_position_duplicates": 0, "quota": {}, "quota_shortfalls": {},
        },
    }

