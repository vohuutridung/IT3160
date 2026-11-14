import json
import sys
from pathlib import Path

import chess
import chess.pgn
import io
import numpy as np
import pytest
import torch

from IT3160.evaluation.agent import PolicyAgent
from IT3160.evaluation.arena import (
    GameRecord, GreedyMaterialPlayer, ModelPlayer, RandomPlayer, openings_from_meta,
    play_game, play_match, select_move, summarize_match, to_pgn,
)
from IT3160.models.resnet import ChessPolicyResNet

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
START = chess.STARTING_FEN


def test_agent_plays_legal_moves():
    torch.manual_seed(0)
    agent = PolicyAgent(ChessPolicyResNet(num_blocks=2, channels=16).eval())
    board = chess.Board()
    policy = agent.policy(board)
    assert len(policy) == 20
    assert abs(sum(p for _, p in policy) - 1.0) < 1e-5
    rng = np.random.default_rng(0)
    for _ in range(30):
        if board.is_game_over():
            break
        move = agent.choose(board, temperature=1.0, rng=rng)
        assert board.is_legal(move)
        board.push(move)
    assert agent.choose(chess.Board()) == policy[0][0]


def tiny_player(name, seed=0):
    torch.manual_seed(seed)
    return ModelPlayer(name, PolicyAgent(ChessPolicyResNet(num_blocks=1, channels=8)))


def test_greedy_takes_material_and_mates():
    rng = np.random.default_rng(0)
    greedy = GreedyMaterialPlayer()
    hanging_queen = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    assert select_move(greedy, hanging_queen, rng) == chess.Move.from_uci("e4d5")
    back_rank = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert select_move(greedy, back_rank, rng) == chess.Move.from_uci("a1a8")


def test_avoid_repetition_skips_claimable_move():
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
        board.push_uci(uci)

    class Fixed:
        name = "fixed"

        def ranked_moves(self, board, rng):
            first = chess.Move.from_uci("f6g8")  # would allow a threefold claim
            return [(first, 0.9)] + [(m, 0.1) for m in board.legal_moves if m != first]

    rng = np.random.default_rng(0)
    assert select_move(Fixed(), board, rng, avoid_repetition=False).uci() == "f6g8"
    assert select_move(Fixed(), board, rng, avoid_repetition=True).uci() != "f6g8"


def test_game_terminates_and_is_legal():
    rng = np.random.default_rng(1)
    game = play_game(RandomPlayer("r1"), RandomPlayer("r2"), START, rng, max_plies=60)
    assert game.plies <= 60
    board = chess.Board(START)
    for uci in game.moves:
        assert board.is_legal(chess.Move.from_uci(uci))
        board.push_uci(uci)
    if game.plies == 60 and game.termination == "max_plies":
        assert game.result == "1/2-1/2"
    mate = play_game(GreedyMaterialPlayer("g"), RandomPlayer("r"),
                     "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", rng)
    assert mate.result == "1-0" and mate.termination == "checkmate" and mate.plies == 1


def test_match_swaps_colours_and_is_deterministic():
    a, b = tiny_player("A", 0), RandomPlayer("B")
    openings = [START, "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"]
    games = play_match(a, b, openings, seed=3, max_plies=40, progress=False)
    assert [(g.white, g.black) for g in games] == [("A", "B"), ("B", "A")] * 2
    again = play_match(a, b, openings, seed=3, max_plies=40, progress=False)
    assert [g.moves for g in games] == [g.moves for g in again]


def test_summary_scores_and_elo():
    def game(i, white, black, result):
        return GameRecord(i, START, white, black, result, "checkmate", 10)

    games = [game(0, "A", "B", "1-0"), game(0, "B", "A", "1/2-1/2"),
             game(1, "A", "B", "0-1"), game(1, "B", "A", "0-1")]
    s = summarize_match(games, "A", "B")
    assert (s["a_wins"], s["draws"], s["a_losses"]) == (2, 1, 1)
    assert s["a_score"] == pytest.approx(0.625)
    assert s["elo_diff"] == pytest.approx(-400 * np.log10(1 / 0.625 - 1))
    assert s["a_as_white"]["score"] == 0.5 and s["a_as_black"]["score"] == 0.75
    assert s["elo_diff_ci95"][0] < s["elo_diff"] < s["elo_diff_ci95"][1]


def test_pgn_roundtrip():
    rng = np.random.default_rng(0)
    record = play_game(RandomPlayer("r1"), RandomPlayer("r2"), START, rng, max_plies=20)
    parsed = chess.pgn.read_game(io.StringIO(to_pgn(record)))
    assert [m.uci() for m in parsed.mainline_moves()] == record.moves
    assert parsed.headers["White"] == "r1" and parsed.headers["Result"] == record.result


def test_openings_from_meta_distinct_and_in_window():
    fens = np.array([chess.Board().fen()] * 3 + [f"8/8/8/8/8/8/{i}P{7 - i}/K6k w - - 0 1"
                                                  for i in range(1, 6)], object)
    plies = np.array([10, 10, 10, 9, 11, 12, 20, 3])
    openings = openings_from_meta(fens, plies, 3, min_ply=8, max_ply=12)
    assert len(set(openings)) == 3
    with pytest.raises(ValueError):
        openings_from_meta(fens, plies, 10)


def test_play_match_script(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    sys.modules.pop("play_match", None)
    import play_match

    torch.manual_seed(0)
    checkpoint = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": ChessPolicyResNet(num_blocks=1, channels=8).state_dict()}, checkpoint)
    openings = tmp_path / "openings.txt"
    openings.write_text(START + "\n", encoding="utf-8")
    out = tmp_path / "match"
    assert play_match.main(["--a", str(checkpoint), "--b", "greedy", "--openings", "1",
                            "--openings-file", str(openings), "--max-plies", "30",
                            "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["games"] == 2
    assert len((out / "games.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert (out / "games.pgn").read_text(encoding="utf-8").count("[Event ") == 2
