
"""
Game and position deduplication.
- Detect duplicate games by complete move sequences.
- Hash canonical board representations.
- Track position occurrences across dataset splits.
- Identify and reject cross-split position conflicts.
"""

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable
import chess
import chess.pgn
from .canonical import encode_board
from .schema import Split


POSITION_HASH_VERSION = "canonical17-legalep-rot180-v2"
GAME_HASH_VERSION = "moves-v1"

SPLIT_BITS = {
    Split.TRAIN: 0b001,
    Split.VALIDATION: 0b010,
    Split.TEST: 0b100,
}


def game_content_hash(game: chess.pgn.Game) -> str:
    """
    Fingerprint a complete game's mainline move sequence.
    Includes the initial position and all mainline moves.
    """

    if not isinstance(game, chess.pgn.Game):
        raise TypeError("game must be a chess.pgn.Game")

    hasher = hashlib.blake2b(digest_size=16)

    hasher.update(GAME_HASH_VERSION.encode("ascii"))
    hasher.update(b"\0")

    initial_fen = game.board().fen()
    hasher.update(initial_fen.encode("ascii"))
    hasher.update(b"\0")

    for move in game.mainline_moves():
        hasher.update(move.uci().encode("ascii"))
        hasher.update(b"\0")

    return hasher.hexdigest()


def position_hash(board: chess.Board) -> str:
    """
    Fingerprint canonical pieces, castling rights and legal en passant.

    A raw FEN can retain an en-passant target even when no legal
    capture exists. Such a target does not distinguish positions
    for leakage checks. Clear it on a copy before encoding; the
    model input encoder and the caller's board remain unchanged.
    """

    if not isinstance(board, chess.Board):
        raise TypeError("board must be a chess.Board")

    canonical_board = board.copy(stack=False)
    if not canonical_board.has_legal_en_passant():
        canonical_board.ep_square = None
    tensor = encode_board(canonical_board)

    hasher = hashlib.blake2b(digest_size=16)

    hasher.update(
        POSITION_HASH_VERSION.encode("ascii")
    )
    hasher.update(b"\0")

    hasher.update(tensor.tobytes(order="C"))

    return hasher.hexdigest()


def _split_bit(split: Split) -> int:
    """Return the bitmask associated with a split."""

    if not isinstance(split, Split):
        raise TypeError("split must be a Split enum")

    return SPLIT_BITS[split]


def _validate_hash(value: str) -> None:
    """Validate a stored fingerprint."""

    if not isinstance(value, str) or not value:
        raise ValueError("Hash must be a non-empty string")


class DeduplicationIndex:
    """
    Persistent SQLite index for game and position deduplication.

    Build stage:
        register_game()
        register_position()
        register_positions()

    Filter stage:
        finalize()
        has_cross_split_conflict()
        is_safe_position()
    """

    _POSITION_UPSERT = """
        INSERT INTO positions (
            position_hash,
            split_mask,
            n_candidates
        )
        VALUES (?, ?, 1)
        ON CONFLICT(position_hash)
        DO UPDATE SET
            split_mask = (
                positions.split_mask | excluded.split_mask
            ),
            n_candidates = positions.n_candidates + 1
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._conn = sqlite3.connect(
            str(self.db_path)
        )

        try:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS games (
                    content_hash TEXT PRIMARY KEY
                ) WITHOUT ROWID;

                CREATE TABLE IF NOT EXISTS positions (
                    position_hash TEXT PRIMARY KEY,
                    split_mask INTEGER NOT NULL,
                    n_candidates INTEGER NOT NULL
                ) WITHOUT ROWID;
                """
            )

            self._set_or_verify_metadata(
                "position_hash_version",
                POSITION_HASH_VERSION,
            )

            self._set_or_verify_metadata(
                "game_hash_version",
                GAME_HASH_VERSION,
            )

            self._conn.execute(
                """
                INSERT OR IGNORE INTO metadata(key, value)
                VALUES ('status', 'building')
                """
            )

            self._conn.commit()
        except BaseException:
            # A failed constructor never reaches __exit__, so release its
            # connection here (including metadata/version check failures).
            self._conn.close()
            raise

    def _set_or_verify_metadata(
        self,
        key: str,
        expected: str,
    ) -> None:
        """Prevent mixing incompatible hash versions."""

        self._conn.execute(
            """
            INSERT OR IGNORE INTO metadata(key, value)
            VALUES (?, ?)
            """,
            (key, expected),
        )

        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()

        if row[0] != expected:
            raise ValueError(
                f"Incompatible index version for {key}: "
                f"{row[0]} != {expected}"
            )

    def _get_status(self) -> str:
        row = self._conn.execute(
            """
            SELECT value FROM metadata
            WHERE key = 'status'
            """
        ).fetchone()

        return row[0]

    def _require_building(self) -> None:
        if self._get_status() != "building":
            raise RuntimeError(
                "Index is finalized; no more inserts allowed"
            )

    def _require_finalized(self) -> None:
        if self._get_status() != "finalized":
            raise RuntimeError(
                "Finalize the index before filtering"
            )

    def register_game(
        self,
        content_hash: str,
    ) -> bool:
        """
        Register a game content fingerprint.

        Returns:
            True if this content is new.
            False if the same content was registered before.
        """

        self._require_building()
        _validate_hash(content_hash)

        cursor = self._conn.execute(
            """
            INSERT OR IGNORE INTO games(content_hash)
            VALUES (?)
            """,
            (content_hash,),
        )

        return cursor.rowcount == 1

    def register_position(
        self,
        fingerprint: str,
        split: Split,
    ) -> None:
        """
        Record the occurrence of a position in one split.
        """

        self._require_building()
        _validate_hash(fingerprint)

        bit = _split_bit(split)

        self._conn.execute(
            self._POSITION_UPSERT,
            (fingerprint, bit),
        )

    def register_positions(
        self,
        positions: Iterable[tuple[str, Split]],
    ) -> None:
        """
        Batch-register (position_hash, split) pairs.

        Accepts a generator and does not materialize all
        records in memory.
        """

        self._require_building()

        def rows():
            for fingerprint, split in positions:
                _validate_hash(fingerprint)
                yield (
                    fingerprint,
                    _split_bit(split),
                )

        self._conn.executemany(
            self._POSITION_UPSERT,
            rows(),
        )

    def commit(self) -> None:
        """Persist pending SQLite transactions."""

        self._conn.commit()

    def finalize(self) -> None:
        """
        Mark the index as complete.

        No additional games or positions can be registered
        after finalization.
        """

        self._conn.execute(
            """
            UPDATE metadata
            SET value = 'finalized'
            WHERE key = 'status'
            """
        )

        self._conn.commit()

    def _get_position_mask(
        self,
        fingerprint: str,
    ) -> int:
        self._require_finalized()
        _validate_hash(fingerprint)

        row = self._conn.execute(
            """
            SELECT split_mask
            FROM positions
            WHERE position_hash = ?
            """,
            (fingerprint,),
        ).fetchone()

        if row is None:
            raise KeyError(
                f"Position not registered: {fingerprint}"
            )

        return row[0]

    def has_cross_split_conflict(
        self,
        fingerprint: str,
    ) -> bool:
        """Check whether a position occurs in multiple splits."""

        mask = self._get_position_mask(fingerprint)

        return (mask & (mask - 1)) != 0

    def is_safe_position(
        self,
        fingerprint: str,
        split: Split,
    ) -> bool:
        """
        Check whether a position belongs exclusively
        to the requested split.

        This method does not remove duplicates occurring
        multiple times within the same split.
        """

        expected_bit = _split_bit(split)
        actual_mask = self._get_position_mask(fingerprint)

        return actual_mask == expected_bit

    def stats(self) -> dict:
        """Return game and position indexing statistics."""

        num_games = self._conn.execute(
            "SELECT COUNT(*) FROM games"
        ).fetchone()[0]

        row = self._conn.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(
                    SUM(
                        (split_mask & (split_mask - 1)) != 0
                    ),
                    0
                ),
                COALESCE(SUM(n_candidates), 0)
            FROM positions
            """
        ).fetchone()

        unique_positions = row[0]
        conflicting_positions = row[1]
        total_candidates = row[2]

        return {
            "unique_game_contents": num_games,
            "unique_position_hashes": unique_positions,
            "cross_split_position_hashes": conflicting_positions,
            "indexed_candidate_occurrences": total_candidates,
            "status": self._get_status(),
            "position_hash_version": POSITION_HASH_VERSION,
            "game_hash_version": GAME_HASH_VERSION,
        }

    def close(self) -> None:
        """Close the SQLite connection."""

        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()

        self._conn.close()

        return False
