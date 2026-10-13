"""Frozen pre-optimization sampler for exact before/after regression checks."""

import hashlib
import random

import chess
import pytest

from IT3160.data.deduplication import game_content_hash
from IT3160.data.pgn_parser import extract_positions
from IT3160.data.phase import classify_phase
from IT3160.data.schema import CandidateRecord, Phase


def legacy_sample_game_candidates(parsed_game, split, *, seed=42,
                                 opening_max_ply=24, endgame_material_max=8):
    """Keep the old FEN roundtrip, record creation and RNG consumption unchanged."""
    content_id = game_content_hash(parsed_game.game)
    reservoirs = {phase: None for phase in Phase}
    counts = {phase: 0 for phase in Phase}
    rngs = {}
    for phase in Phase:
        payload = f"sampling-v1|{seed}|{content_id}|{phase.value}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        rngs[phase] = random.Random(int.from_bytes(digest, byteorder="big", signed=False))
    for raw in extract_positions(parsed_game):
        phase = classify_phase(chess.Board(raw["fen"]), raw["ply"],
                               opening_max_ply=opening_max_ply,
                               endgame_material_max=endgame_material_max)
        counts[phase] += 1
        candidate = CandidateRecord(
            game_id=content_id, source_month=parsed_game.source_month, split=split,
            phase=phase, ply=raw["ply"], fen=raw["fen"], move_uci=raw["move_uci"],
            white_elo=raw["white_elo"], black_elo=raw["black_elo"], result=raw["result"],
        )
        if reservoirs[phase] is None:
            reservoirs[phase] = candidate
            continue
        if rngs[phase].randrange(counts[phase]) == 0:
            reservoirs[phase] = candidate
    return [reservoirs[phase] for phase in Phase if reservoirs[phase] is not None]


@pytest.fixture
def legacy_sampler():
    return legacy_sample_game_candidates
