"""
Per-game phase sampling for the chess-policy dataset.

Selects at most one position per phase from a single game
using deterministic reservoir sampling. Dataset-level quota
balancing is handled by the pipeline, not this module.
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterator

import chess

from .deduplication import game_content_hash
from .phase import (
    ENDGAME_MATERIAL_MAX,
    OPENING_MAX_PLY,
    classify_phase,
)
from .pgn_parser import ParsedGame, extract_positions
from .schema import CandidateRecord, Phase, Split


PHASE_ORDER = (
    Phase.OPENING,
    Phase.MIDDLEGAME,
    Phase.ENDGAME,
)


def _derive_phase_seed(
    seed: int,
    game_id: str,
    phase: Phase,
) -> int:
    """
    Derive a stable RNG seed for one (game, phase) reservoir.

    The result depends only on the global seed, content-based
    game ID, and phase label — never on processing order.
    """

    payload = (
        f"sampling-v1|{seed}|{game_id}|{phase.value}"
    ).encode("utf-8")

    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def sample_game_candidates(
    parsed_game: ParsedGame,
    split: Split,
    *,
    seed: int = 42,
    opening_max_ply: int = OPENING_MAX_PLY,
    endgame_material_max: int = ENDGAME_MATERIAL_MAX,
    max_samples_per_phase: int = 1,
) -> list[CandidateRecord]:
    """
    Sample at most one CandidateRecord per phase from a game.

    Args:
        parsed_game:
            A validated PGN game wrapper.
        split:
            Dataset split already assigned for this game.
            Must be computed from the content hash game ID.
        seed:
            Global sampling seed.
        opening_max_ply:
            Opening boundary passed to classify_phase().
        endgame_material_max:
            Endgame material threshold.
        max_samples_per_phase:
            Must be 1. Reserved for future extension.

    Returns:
        Zero to three CandidateRecord objects, ordered as
        Opening, Middlegame, Endgame when present.

    Notes:
        CandidateRecord.game_id is the content hash of the
        full mainline (initial FEN + UCI moves), not the
        metadata-based ParsedGame.game_id.
    """

    if not isinstance(parsed_game, ParsedGame):
        raise TypeError("parsed_game must be a ParsedGame")

    if not isinstance(split, Split):
        raise TypeError("split must be a Split enum")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if isinstance(max_samples_per_phase, bool) or not isinstance(
        max_samples_per_phase, int
    ):
        raise TypeError("max_samples_per_phase must be an integer")

    if max_samples_per_phase != 1:
        raise ValueError(
            "Only max_samples_per_phase=1 is supported"
        )

    content_id = game_content_hash(parsed_game.game)

    reservoirs: dict[Phase, CandidateRecord | None] = {
        phase: None for phase in PHASE_ORDER
    }
    counts: dict[Phase, int] = {
        phase: 0 for phase in PHASE_ORDER
    }
    rngs: dict[Phase, random.Random] = {
        phase: random.Random(
            _derive_phase_seed(seed, content_id, phase)
        )
        for phase in PHASE_ORDER
    }

    white_elo: int | None = None
    black_elo: int | None = None
    result: str | None = None
    metadata_loaded = False

    for raw in extract_positions(parsed_game):
        if not metadata_loaded:
            white_elo = raw["white_elo"]
            black_elo = raw["black_elo"]
            result = raw["result"]
            metadata_loaded = True

        board = chess.Board(raw["fen"])
        phase = classify_phase(
            board,
            raw["ply"],
            opening_max_ply=opening_max_ply,
            endgame_material_max=endgame_material_max,
        )

        counts[phase] += 1
        seen = counts[phase]

        candidate = CandidateRecord(
            game_id=content_id,
            source_month=parsed_game.source_month,
            split=split,
            phase=phase,
            ply=raw["ply"],
            fen=raw["fen"],
            move_uci=raw["move_uci"],
            white_elo=white_elo,
            black_elo=black_elo,
            result=result,
        )

        if reservoirs[phase] is None:
            reservoirs[phase] = candidate
            continue

        # Algorithm R: replace with probability 1/seen.
        if rngs[phase].randrange(seen) == 0:
            reservoirs[phase] = candidate

    return [
        reservoirs[phase]
        for phase in PHASE_ORDER
        if reservoirs[phase] is not None
    ]


def iter_game_candidates(
    parsed_games: Iterator[ParsedGame],
    *,
    assign_split_fn,
    seed: int = 42,
    opening_max_ply: int = OPENING_MAX_PLY,
    endgame_material_max: int = ENDGAME_MATERIAL_MAX,
) -> Iterator[CandidateRecord]:
    """
    Stream CandidateRecords from an iterator of ParsedGames.

    ``assign_split_fn`` must accept the content-hash game ID
    and return a Split. Typical usage:

        lambda game_id: assign_split(game_id, seed=seed, ratios=...)
    """

    for parsed_game in parsed_games:
        content_id = game_content_hash(parsed_game.game)
        split = assign_split_fn(content_id)

        yield from sample_game_candidates(
            parsed_game,
            split,
            seed=seed,
            opening_max_ply=opening_max_ply,
            endgame_material_max=endgame_material_max,
        )
