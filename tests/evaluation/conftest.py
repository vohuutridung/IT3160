"""A tiny exported split built from real games, laid out exactly like tensor_export output."""

import json

import chess
import h5py
import numpy as np
import pytest

from IT3160.data.canonical import encode_board
from IT3160.data.phase import classify_phase
from IT3160.data.pipeline import encode_candidate
from IT3160.data.schema import ACTION_SPACE_SIZE, BOARD_SHAPE, CandidateRecord, Split
from IT3160.data.storage import write_shard
from IT3160.data.tensor_export import PHASE_IDS

GAMES = [
    # Includes castling, captures and checks.
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1g1 f8c5 d2d3 d7d6 c2c3 e8g8 b2b4 c5b6 a2a4 a7a6 "
    "c1g5 h7h6 g5h4 g7g5 f3g5 h6g5 h4g5 c8e6 c4e6 f7e6 d1f3",
    # Reaches a promotion.
    "a2a4 b7b5 a4b5 a7a6 b5a6 c8b7 a6b7 g8f6 b7a8q",
]


def _records():
    records = []
    for game_index, game in enumerate(GAMES):
        board = chess.Board()
        for uci in game.split():
            move = chess.Move.from_uci(uci)
            candidate = CandidateRecord(
                game_id=f"{game_index:032x}", source_month="2021-12", split=Split.TEST,
                phase=classify_phase(board), ply=board.ply(), fen=board.fen(), move_uci=uci,
                white_elo=2500 + 100 * game_index, black_elo=2450, result="1-0",
            )
            records.append(encode_candidate(candidate))
            board.push(move)
    return records


@pytest.fixture
def exported_split(tmp_path):
    """Return (h5_path, processed_dir, records)."""
    records = _records()
    processed_dir = tmp_path / "processed"
    remote = "processed/run/test/part-00000.parquet"
    shard_path = processed_dir / remote
    shard_path.parent.mkdir(parents=True)
    info = write_shard(records, shard_path, kind="processed")
    n = len(records)
    h5_path = tmp_path / "test.h5"
    metadata = {
        "repo_id": "local/test", "revision": "0" * 40, "split": "test", "rows": n,
        "source_shards": [{"remote_path": remote, "rows": n, "sha256": info.sha256,
                           "size_bytes": info.size_bytes}],
    }
    with h5py.File(h5_path, "w") as out:
        out.attrs["metadata"] = json.dumps(metadata)
        x = np.stack([encode_board(chess.Board(r.fen)) for r in records]).astype(np.uint8)
        mask = np.zeros((n, ACTION_SPACE_SIZE), bool)
        for i, r in enumerate(records):
            mask[i, list(r.legal_action_ids)] = True
        out.create_dataset("x", data=x, chunks=(4, *BOARD_SHAPE), compression="gzip")
        out.create_dataset("y", data=np.array([r.action_id for r in records], np.int64))
        out.create_dataset("legal_mask", data=mask, chunks=(4, ACTION_SPACE_SIZE), compression="gzip")
        out.create_dataset("phase", data=np.array([PHASE_IDS[r.phase.value] for r in records], np.uint8))
    return h5_path, processed_dir, records
