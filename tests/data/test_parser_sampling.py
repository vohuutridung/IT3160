"""Self-contained archive fixtures and checks of raw pre-move sampling."""

from __future__ import annotations

from collections import Counter
import io
from pathlib import Path
import random
import zipfile

import chess
import chess.pgn
import pytest

from IT3160.data.deduplication import game_content_hash
from IT3160.data.pgn_parser import (
    ParsedGame,
    ParserStats,
    extract_positions,
    is_valid_game,
    iter_valid_games,
    make_game_id,
)
from IT3160.data.phase import classify_phase, material_score
from IT3160.data import sampling
from IT3160.data.sampling import iter_game_candidates, sample_game_candidates
from IT3160.data.schema import Phase, Split


def pgn(moves: str = "1. e4 e5 2. Nf3 Nc6 1-0", **headers: str) -> str:
    tags = {
        "Event": "test",
        "Site": "local",
        "Date": "2021.12.01",
        "White": "White player",
        "Black": "Black player",
        "Result": "1-0",
    }
    tags.update(headers)
    return "\n".join(f'[{name} "{value}"]' for name, value in tags.items()) + (
        "\n\n" + moves + "\n\n"
    )


def archive(tmp_path: Path, *members: tuple[str, str | bytes]) -> Path:
    destination = tmp_path / "games.zip"
    with zipfile.ZipFile(destination, "w") as zipped:
        for name, text in members:
            zipped.writestr(name, text)
    return destination


def wrap_game(game: chess.pgn.Game, month: str = "2021-12") -> ParsedGame:
    return ParsedGame(make_game_id(game), month, game)


def parse_game(text: str) -> chess.pgn.Game:
    game = chess.pgn.read_game(io.StringIO(text))
    assert game is not None
    return game


@pytest.fixture
def short_game() -> ParsedGame:
    return wrap_game(parse_game(pgn()))


@pytest.fixture(scope="module")
def all_phase_game() -> ParsedGame:
    # Generate a legal, reproducible game locally; no external PGNs/downloads.
    # Prefer captures after the opening so the fixture includes G <= 8.
    for attempt in range(100):
        rng = random.Random(attempt)
        board = chess.Board()
        for _ in range(180):
            legal = list(board.legal_moves)
            if not legal:
                break
            captures = [move for move in legal if board.is_capture(move)]
            pool = captures if board.ply() > 30 and captures else legal
            board.push(rng.choice(pool))
            if board.ply() > 50 and material_score(board) <= 8:
                # Include a human action from the reached endgame board.
                continuations = list(board.legal_moves)
                if continuations:
                    board.push(rng.choice(continuations))
                    game = chess.pgn.Game.from_board(board)
                    game.headers["Result"] = "1-0"
                    parsed = wrap_game(game)
                    assert is_valid_game(game)
                    assert {
                        classify_phase(chess.Board(raw["fen"]), raw["ply"])
                        for raw in extract_positions(parsed)
                    } == set(Phase)
                    return parsed
                break
    pytest.fail("Could not construct the deterministic three-phase fixture")


def test_streams_all_pgn_members_in_sorted_order_and_skips_other_files(
    tmp_path, monkeypatch
):
    path = archive(
        tmp_path,
        ("z.pgn", pgn(Site="second")),
        ("notes.txt", "not a PGN"),
        ("nested/A.PGN", b"\xef\xbb\xbf" + pgn(Site="first").encode()),
        ("directory/", ""),
    )

    def do_not_read_archive_member(*args, **kwargs):
        raise AssertionError("ZIP members must be streamed with open(), not read()")

    monkeypatch.setattr(zipfile.ZipFile, "read", do_not_read_archive_member)
    games = list(iter_valid_games(path, "2021-12"))
    assert [game.game.headers["Site"] for game in games] == ["first", "second"]
    assert all(game.source_month == "2021-12" for game in games)
    assert len({game.game_id for game in games}) == 2


def test_duplicate_zip_member_names_are_each_read_once(tmp_path):
    with pytest.warns(UserWarning, match="Duplicate name"):
        path = archive(
            tmp_path,
            ("games.pgn", pgn(Site="first entry")),
            ("games.pgn", pgn(Site="second entry")),
        )
    assert [
        game.game.headers["Site"] for game in iter_valid_games(path, "2021-12")
    ] == ["first entry", "second entry"]


def test_valid_game_limit_counts_accepted_games_and_stops_reading(tmp_path, monkeypatch):
    path = archive(
        tmp_path,
        ("games.pgn", pgn("*") + pgn(Site="first") + pgn(Site="unread")),
    )
    original = chess.pgn.read_game
    calls = 0

    def read_game(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(chess.pgn, "read_game", read_game)
    stats = ParserStats()
    games = list(iter_valid_games(path, "2021-12", max_games=1, stats=stats))
    assert [game.game.headers["Site"] for game in games] == ["first"]
    assert calls == 2
    assert stats.games_seen == 2
    assert stats.games_accepted == 1


def test_zero_limit_does_not_open_nonexistent_source(tmp_path):
    stats = ParserStats()
    assert list(iter_valid_games(tmp_path / "absent.zip", "2021-12", 0, stats=stats)) == []
    assert stats.to_dict()["games_seen"] == 0


@pytest.mark.parametrize("month", ["2021-00", "2021-13", "2021-1", "21-12", "2021-12\n", "٢٠٢١-12", "", None, 202112])
def test_invalid_month_fails_before_archive_access(tmp_path, month):
    with pytest.raises(ValueError, match="source_month"):
        list(iter_valid_games(tmp_path / "absent.zip", month))


@pytest.mark.parametrize("limit", [True, False, 1.0, "2"])
def test_game_limit_requires_integer(tmp_path, limit):
    with pytest.raises(TypeError, match="max_games"):
        list(iter_valid_games(tmp_path / "absent.zip", "2021-12", limit))


def test_negative_limit_and_invalid_stats_fail_before_archive_access(tmp_path):
    with pytest.raises(ValueError, match="max_games"):
        list(iter_valid_games(tmp_path / "absent.zip", "2021-12", -1))
    with pytest.raises(TypeError, match="stats"):
        list(iter_valid_games(tmp_path / "absent.zip", "2021-12", stats={}))


def test_zip_with_no_pgn_and_invalid_zip_raise_clear_errors(tmp_path):
    with pytest.raises(ValueError, match="No PGN files"):
        list(iter_valid_games(archive(tmp_path, ("notes.txt", "text")), "2021-12"))
    path = tmp_path / "corrupt.zip"
    path.write_bytes(b"not a ZIP archive")
    with pytest.raises(zipfile.BadZipFile):
        list(iter_valid_games(path, "2021-12"))


@pytest.mark.parametrize(
    "text",
    [
        pgn("1. e4 e5"),  # Truncated movetext with a complete Result header.
        pgn("1. e4 e5 0-1"),  # Header/movetext result disagreement.
        pgn("1. e4 e5 *"),
        pgn("1. e4 e5 1-0 0-1"),
        pgn("1. e4 e5 1-0 2. Nf3"),
        pgn("1. e4 e5 2. Qh8 1-0"),  # Recognized but illegal SAN.
        pgn("1. e4 broken e5 1-0"),  # Silently ignored by default python-chess.
        pgn("1. e4 e5 1-0 {comment never closed"),
        pgn("1. e4 (1. d4 d5 1-0"),
        pgn("1. e4 ) e5 1-0"),
        pgn("1. e4 e5 1-0").replace('[Site "local"]', '[Site "local"'),
        pgn("1-0"),  # No moves.
        pgn("1. e4 e5 *", Result="*"),
        pgn(Variant="Chess960"),
        pgn(Variant="Atomic"),
        pgn("1. Kf2 1-0", SetUp="1", FEN="8/8/8/8/8/8/4K3/7k w - - 0 1"),
        pgn(FEN="not a board"),
    ],
)
def test_rejects_invalid_games_without_losing_next_valid_game(tmp_path, text):
    # Put incomplete comment/variation in a separate member: malformed comment
    # content cannot establish the boundary of the next game within one member.
    path = archive(tmp_path, ("a_bad.pgn", text), ("z_good.pgn", pgn(Site="good")))
    stats = ParserStats()
    assert [game.game.headers["Site"] for game in iter_valid_games(path, "2021-12", stats=stats)] == ["good"]
    assert stats.games_seen == 2
    assert stats.games_accepted == 1
    assert sum(stats.rejection_reasons.values()) == 1


@pytest.mark.parametrize("variant", [None, "Standard", "Chess", "standard", "chess"])
def test_standard_variant_headers_are_accepted(tmp_path, variant):
    headers = {} if variant is None else {"Variant": variant}
    assert len(list(iter_valid_games(archive(tmp_path, ("g.pgn", pgn(**headers))), "2021-12"))) == 1


def test_explicit_standard_initial_fen_is_accepted(tmp_path):
    text = pgn(SetUp="1", FEN=chess.STARTING_FEN)
    assert len(list(iter_valid_games(archive(tmp_path, ("g.pgn", text)), "2021-12"))) == 1


@pytest.mark.parametrize("termination", ["Normal", "Time forfeit", "Resignation", "unknown"])
def test_completed_time_losses_and_resignations_are_not_filtered(tmp_path, termination):
    stats = ParserStats()
    text = pgn(Termination=termination, WhiteElo="0", BlackElo="?")
    games = list(iter_valid_games(archive(tmp_path, ("g.pgn", text)), "2021-12", stats=stats))
    assert len(games) == 1
    assert games[0].game.headers["Termination"] == termination
    assert stats.to_dict()["terminations"] == {termination: 1}
    assert stats.to_dict()["results"] == {"1-0": 1}


@pytest.mark.parametrize("result", ["1-0", "0-1", "1/2-1/2"])
def test_completed_result_is_preserved_for_every_position(tmp_path, result):
    text = pgn(f"1. e4 e5 {result}", Result=result)
    games = list(iter_valid_games(archive(tmp_path, ("g.pgn", text)), "2021-12"))
    assert [raw["result"] for raw in extract_positions(games[0])] == [result, result]


def test_annotations_and_multiline_comments_do_not_change_mainline(tmp_path):
    text = pgn(
        "1. e4! {arbitrary text: broken $ and\n\n"
        "braces-free commentary} e5 (1... c5 2. Nf3) "
        "2. Nf3 $1 Nc6 ; semicolon comment\n1-0"
    )
    games = list(iter_valid_games(archive(tmp_path, ("g.pgn", text)), "2021-12"))
    assert len(games) == 1
    assert [raw["move_uci"] for raw in extract_positions(games[0])] == ["e2e4", "e7e5", "g1f3", "b8c6"]


def test_check_and_checkmate_suffixes_are_accepted(tmp_path):
    text = pgn("1. f3 e5 2. g4 Qh4# 0-1", Result="0-1")
    games = list(iter_valid_games(archive(tmp_path, ("g.pgn", text)), "2021-12"))
    assert len(games) == 1
    assert games[0].game.end().board().is_checkmate()


@pytest.mark.parametrize(
    "bad_game",
    [
        pgn("1. e4 e5"),
        pgn("1. e4 junk e5 1-0"),
        pgn("1. e4 e5 0-1"),
        pgn("1. e4 e5 2. Qh8 1-0"),
        pgn("1-0"),
    ],
)
def test_rejection_resumes_at_next_game_in_the_same_member(tmp_path, bad_game):
    path = archive(tmp_path, ("g.pgn", bad_game + pgn(Site="next good game")))
    assert [game.game.headers["Site"] for game in iter_valid_games(path, "2021-12")] == ["next good game"]


def test_pgn_without_headers_can_use_an_explicit_movetext_result(tmp_path):
    path = archive(tmp_path, ("g.pgn", "1. e4 e5 1-0\n"))
    games = list(iter_valid_games(path, "2021-12"))
    assert len(games) == 1
    assert games[0].game.headers["Result"] == "1-0"


def test_parser_stats_are_reusable_and_serializable(tmp_path):
    stats = ParserStats()
    path = archive(tmp_path, ("g.pgn", pgn(Termination="Normal") + pgn("1-0")))
    list(iter_valid_games(path, "2021-12", stats=stats))
    list(iter_valid_games(path, "2022-01", stats=stats))
    assert stats.to_dict() == {
        "games_seen": 4,
        "games_accepted": 2,
        "games_rejected": 2,
        "rejection_reasons": {"no_moves": 2},
        "results": {"1-0": 4},
        "terminations": {"Normal": 2, "unknown": 2},
        "accepted_results": {"1-0": 2},
        "rejected_results": {"1-0": 2},
        "accepted_terminations": {"Normal": 2},
        "rejected_terminations": {"unknown": 2},
    }
    snapshot = stats.to_dict()
    snapshot["results"]["1-0"] = 999
    assert stats.results["1-0"] == 4


def test_extraction_has_exact_pre_move_board_ply_and_metadata():
    game = wrap_game(parse_game(pgn(WhiteElo=" 2531 ", BlackElo="-1")))
    records = list(extract_positions(game))
    assert [record["ply"] for record in records] == [0, 1, 2, 3]
    assert records[0]["fen"] == chess.STARTING_FEN
    board = chess.Board()
    for record in records:
        assert record["fen"] == board.fen()
        assert record["game_id"] == game.game_id
        assert record["source_month"] == "2021-12"
        assert record["white_elo"] == 2531
        assert record["black_elo"] is None
        move = chess.Move.from_uci(record["move_uci"])
        assert board.is_legal(move)
        board.push(move)
    assert records[-1]["fen"] != board.fen()


@pytest.mark.parametrize("elo", [None, "?", "not an Elo", "-5"])
def test_unavailable_elo_is_none(elo):
    headers = {} if elo is None else {"WhiteElo": elo}
    game = wrap_game(parse_game(pgn(**headers)))
    assert {record["white_elo"] for record in extract_positions(game)} == {None}


def test_validation_and_extraction_reject_manually_injected_illegal_move(short_game):
    short_game.game.end().add_main_variation(chess.Move.from_uci("a1a8"))
    assert not is_valid_game(short_game.game)
    with pytest.raises(ValueError, match="Illegal move"):
        list(extract_positions(short_game))


def test_metadata_game_id_is_stable_but_content_id_ignores_metadata(short_game):
    same_moves = parse_game(pgn(White="Other player", Site="different"))
    assert make_game_id(short_game.game) == make_game_id(parse_game(pgn()))
    assert make_game_id(short_game.game) != make_game_id(same_moves)
    assert game_content_hash(short_game.game) == game_content_hash(same_moves)


def test_sampling_selects_one_legal_pre_move_position_per_phase(all_phase_game):
    sampled = sample_game_candidates(all_phase_game, Split.TRAIN)
    assert [candidate.phase for candidate in sampled] == list(Phase)
    assert Counter(candidate.phase for candidate in sampled) == Counter({phase: 1 for phase in Phase})
    records = {raw["ply"]: raw for raw in extract_positions(all_phase_game)}
    for candidate in sampled:
        raw = records[candidate.ply]
        assert (candidate.fen, candidate.move_uci) == (raw["fen"], raw["move_uci"])
        board = chess.Board(candidate.fen)
        assert board.is_valid()
        assert board.is_legal(chess.Move.from_uci(candidate.move_uci))
        assert candidate.phase == classify_phase(board, candidate.ply)
        assert candidate.game_id == game_content_hash(all_phase_game.game)
        assert candidate.split == Split.TRAIN


def test_sampling_repeatability_and_independence_of_metadata_and_stream_order(all_phase_game, short_game):
    expected = sample_game_candidates(all_phase_game, Split.TEST, seed=71)
    assert sample_game_candidates(all_phase_game, Split.TEST, seed=71) == expected
    duplicate = parse_game(str(all_phase_game.game))
    duplicate.headers["Site"] = "metadata changed"
    duplicate.headers["White"] = "another name"
    wrapped = wrap_game(duplicate)
    assert wrapped.game_id != all_phase_game.game_id
    assert sample_game_candidates(wrapped, Split.TEST, seed=71) == expected
    forward = list(iter_game_candidates(iter([all_phase_game, short_game]), assign_split_fn=lambda _: Split.TEST, seed=71))
    backward = list(iter_game_candidates(iter([short_game, all_phase_game]), assign_split_fn=lambda _: Split.TEST, seed=71))
    assert sorted(forward, key=lambda row: (row.game_id, row.phase.value)) == sorted(backward, key=lambda row: (row.game_id, row.phase.value))


def test_reservoir_can_select_every_short_game_position_with_no_first_last_bias(short_game):
    # Check observed selection across deterministic seeds against the intended
    # uniform reservoir behavior, using wide bounds (not a flaky random test).
    counts = Counter(
        sample_game_candidates(short_game, Split.TRAIN, seed=seed)[0].ply
        for seed in range(1000)
    )
    assert set(counts) == {0, 1, 2, 3}
    assert all(190 <= count <= 310 for count in counts.values())


def test_opening_boundary_includes_ply_24_and_excludes_ply_25(all_phase_game):
    opening_plys = {
        raw["ply"]
        for raw in extract_positions(all_phase_game)
        if classify_phase(chess.Board(raw["fen"]), raw["ply"]) == Phase.OPENING
    }
    assert opening_plys == set(range(25))
    sampled_opening_plys = {
        sample_game_candidates(all_phase_game, Split.TRAIN, seed=seed)[0].ply
        for seed in range(1000)
    }
    assert sampled_opening_plys == set(range(25))


def test_stream_split_assignment_receives_content_hash(short_game):
    observed = []

    def assign(game_id):
        observed.append(game_id)
        return Split.VALIDATION

    sampled = list(iter_game_candidates(iter([short_game]), assign_split_fn=assign))
    assert observed == [game_content_hash(short_game.game)]
    assert sampled[0].split == Split.VALIDATION


@pytest.mark.parametrize("seed", [True, False, 1.5, "42", None])
def test_sampling_seed_requires_integer(short_game, seed):
    with pytest.raises(TypeError, match="seed"):
        sample_game_candidates(short_game, Split.TRAIN, seed=seed)


@pytest.mark.parametrize("value", [True, False, 1.0, "1"])
def test_per_phase_sample_limit_requires_integer(short_game, value):
    with pytest.raises(TypeError, match="max_samples_per_phase"):
        sample_game_candidates(short_game, Split.TRAIN, max_samples_per_phase=value)


@pytest.mark.parametrize("value", [0, 2, -1])
def test_only_one_sample_per_phase_is_supported(short_game, value):
    with pytest.raises(ValueError, match="max_samples_per_phase=1"):
        sample_game_candidates(short_game, Split.TRAIN, max_samples_per_phase=value)


def test_sampling_types_and_negative_thresholds_are_checked(short_game):
    with pytest.raises(TypeError, match="parsed_game"):
        sample_game_candidates(short_game.game, Split.TRAIN)
    with pytest.raises(TypeError, match="split"):
        sample_game_candidates(short_game, "train")
    with pytest.raises(ValueError, match="opening_max_ply"):
        sample_game_candidates(short_game, Split.TRAIN, opening_max_ply=-1)
    with pytest.raises(ValueError, match="endgame_material_max"):
        sample_game_candidates(short_game, Split.TRAIN, endgame_material_max=-1)


@pytest.mark.parametrize("seed", [-7, 0, 1, 42, 71, 2**64])
@pytest.mark.parametrize("thresholds", [(24, 8), (0, 64), (5, 0), (1000, 8)])
def test_optimized_sampling_matches_legacy_records_and_order(
    short_game, all_phase_game, legacy_sampler, seed, thresholds
):
    options = dict(seed=seed, opening_max_ply=thresholds[0], endgame_material_max=thresholds[1])
    for parsed in (short_game, all_phase_game):
        for split in Split:
            assert sample_game_candidates(parsed, split, **options) == legacy_sampler(
                parsed, split, **options
            )


@pytest.mark.parametrize("fen,move", [
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
    ("k7/8/8/4KPp1/8/8/8/8 w - g6 0 1", "f5g6"),
    ("4k3/P7/8/8/8/8/8/4K3 w - - 0 14", "a7a8n"),
    ("4k3/P7/8/8/8/8/8/4K3 w - - 0 14", "a7a8q"),
    ("4k3/8/8/8/8/8/p7/4K3 b - - 0 14", "a2a1n"),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "e7e5"),
])
def test_snapshot_fen_matches_legacy_for_special_moves(legacy_sampler, fen, move):
    board = chess.Board(fen)
    assert board.is_legal(chess.Move.from_uci(move))
    game = chess.pgn.Game()
    game.setup(board)
    game.headers["Result"] = "1-0"
    game.add_main_variation(chess.Move.from_uci(move))
    parsed = wrap_game(game)
    actual = sample_game_candidates(parsed, Split.TRAIN)
    assert actual == legacy_sampler(parsed, Split.TRAIN)
    assert actual[0].fen == board.fen()
    assert actual[0].move_uci == move


def test_only_final_positions_are_serialized_and_materialized(all_phase_game, monkeypatch):
    fen_calls = []
    created = []
    real_fen = chess.Board.fen
    real_record = sampling.CandidateRecord

    def fen(board, *args, **kwargs):
        fen_calls.append(board.ply())
        return real_fen(board, *args, **kwargs)

    def record(**kwargs):
        created.append(kwargs)
        return real_record(**kwargs)

    monkeypatch.setattr(chess.Board, "fen", fen)
    monkeypatch.setattr(sampling, "CandidateRecord", record)
    actual = sample_game_candidates(all_phase_game, Split.TRAIN)
    assert len(actual) == len(created) == 3
    # One FEN for the game content hash, then only the three selected positions.
    assert len(fen_calls) == 4


def test_iter_sampler_reuses_hash_without_double_computation(short_game, monkeypatch):
    calls = []
    real_hash = sampling.game_content_hash

    def content_hash(game):
        calls.append(game)
        return real_hash(game)

    monkeypatch.setattr(sampling, "game_content_hash", content_hash)
    list(iter_game_candidates(iter([short_game]), assign_split_fn=lambda _: Split.TRAIN))
    assert calls == [short_game.game]


def test_public_sampling_rechecks_parser_output_after_mutation(tmp_path):
    parsed = next(iter_valid_games(archive(tmp_path, ("valid.pgn", pgn())), "2021-12"))
    parsed.game.end().add_main_variation(chess.Move.from_uci("a1a8"))
    with pytest.raises(ValueError, match="Illegal move"):
        sample_game_candidates(parsed, Split.TRAIN)
    with pytest.raises(ValueError, match="Illegal move"):
        list(iter_game_candidates(iter([parsed]), assign_split_fn=lambda _: Split.TRAIN))


def test_public_content_hash_is_not_cached_on_mutable_game(short_game):
    original_id = sample_game_candidates(short_game, Split.TRAIN)[0].game_id
    short_game.game.end().add_main_variation(chess.Move.from_uci("f1b5"))
    result = sample_game_candidates(short_game, Split.TRAIN)
    assert result[0].game_id != original_id
    assert result[0].game_id == game_content_hash(short_game.game)
