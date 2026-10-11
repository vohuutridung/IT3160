from dataclasses import dataclass, asdict
from enum import Enum
import re
from typing import Optional


BOARD_SHAPE = (17, 8, 8)
ACTION_SPACE_SIZE = 4672


def _validate_text(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} cannot be empty")


def _validate_nonnegative_integer(name: str, value: int) -> None:
    # bool is an int subclass, but it is not a valid count or action label.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_common(record: "CandidateRecord | ProcessedRecord") -> None:
    """Validate field types here; chess semantics belong in validation.py."""
    for name in ("game_id", "source_month", "fen", "move_uci"):
        _validate_text(name, getattr(record, name))

    if not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", record.source_month):
        raise ValueError("source_month must use YYYY-MM format")

    if not isinstance(record.phase, Phase):
        raise TypeError("phase must be a Phase enum")
    if not isinstance(record.split, Split):
        raise TypeError("split must be a Split enum")

    _validate_nonnegative_integer("ply", record.ply)
    for name in ("white_elo", "black_elo"):
        value = getattr(record, name)
        if value is not None:
            _validate_nonnegative_integer(name, value)

    if record.result is not None:
        _validate_text("result", record.result)
        if record.result not in {"1-0", "0-1", "1/2-1/2", "*"}:
            raise ValueError("result must be a conventional PGN result")

class Phase(str, Enum):
    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"

class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

@dataclass(frozen=True, slots=True)
class CandidateRecord:
    game_id: str
    source_month: str

    split: Split
    phase: Phase

    ply: int
    fen: str
    move_uci: str

    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    result: Optional[str] = None

    def __post_init__(self):
        _validate_common(self)

    def to_dict(self) -> dict:
        record = asdict(self)
        record["phase"] = self.phase.value
        record["split"] = self.split.value
        return record

@dataclass(frozen=True, slots=True)
class ProcessedRecord:
    """
    A processed position ready for the dataset loader.
    """
    game_id: str
    source_month: str

    split: Split
    phase: Phase

    ply: int
    fen: str
    move_uci: str

    position_hash: str

    action_id: int
    legal_action_ids: tuple[int, ...]

    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    result: Optional[str] = None

    def __post_init__(self):
        _validate_common(self)
        _validate_text("position_hash", self.position_hash)
        _validate_nonnegative_integer("action_id", self.action_id)

        if not 0 <= self.action_id < ACTION_SPACE_SIZE:
            raise ValueError("Invalid action_id")

        if not isinstance(self.legal_action_ids, tuple):
            raise TypeError("legal_action_ids must be a tuple of integers")

        if not self.legal_action_ids:
            raise ValueError("legal_action_ids cannot be empty")

        for action in self.legal_action_ids:
            _validate_nonnegative_integer("legal action ID", action)

        if len(set(self.legal_action_ids)) != len(self.legal_action_ids):
            raise ValueError("Duplicate legal action IDs")

        if any(
            not 0 <= action < ACTION_SPACE_SIZE
            for action in self.legal_action_ids
        ):
            raise ValueError("Legal action ID out of range")

        if self.action_id not in self.legal_action_ids:
            raise ValueError("Target action must be legal")

    def to_dict(self) -> dict:
        record = asdict(self)
        record["phase"] = self.phase.value
        record["split"] = self.split.value
        record["legal_action_ids"] = list(self.legal_action_ids)
        return record
