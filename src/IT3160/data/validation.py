"""
Validation utilities for candidate and processed dataset rows.

Performs:
- Per-record integrity checks (FEN, move, phase, encoding).
- Streaming whole-dataset checks (cross-split leakage, quotas).

Designed for large datasets: issues are capped and aggregate
statistics are collected via a temporary SQLite index.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import chess

from .action_encoding import (
    decode_move,
    encode_move,
    legal_action_ids,
)
from .deduplication import position_hash
from .phase import (
    ENDGAME_MATERIAL_MAX,
    OPENING_MAX_PLY,
    classify_phase,
)
from .progress import Progress, log
from .schema import (
    ACTION_SPACE_SIZE,
    CandidateRecord,
    Phase,
    ProcessedRecord,
    Split,
)
from .split import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    SplitRatios,
    assign_split,
)


# ---------------------------------------------------------
# Issue / report types
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One concrete validation failure."""

    code: str
    message: str
    game_id: Optional[str] = None
    split: Optional[str] = None
    phase: Optional[str] = None
    ply: Optional[int] = None
    position_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ValidationReport:
    """Aggregated validation outcome."""

    kind: str
    checked: int = 0
    error_count: int = 0
    warning_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    max_issues: int = 100

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def add_error(self, issue: ValidationIssue) -> None:
        self.error_count += 1
        if len(self.issues) < self.max_issues:
            self.issues.append(issue)

    def add_warning(self, issue: ValidationIssue) -> None:
        self.warning_count += 1
        if len(self.issues) < self.max_issues:
            self.issues.append(issue)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "ok": self.ok,
            "checked": self.checked,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_dict() for issue in self.issues],
            "stats": self.stats,
            "max_issues": self.max_issues,
            "issues_truncated": (
                self.error_count + self.warning_count
            ) > len(self.issues),
        }


# ---------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------

def _issue(
    code: str,
    message: str,
    record: CandidateRecord | ProcessedRecord,
    *,
    position_hash_value: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        game_id=record.game_id,
        split=record.split.value,
        phase=record.phase.value,
        ply=record.ply,
        position_hash=position_hash_value,
    )


def _load_board(fen: str) -> chess.Board:
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError("Board fails chess.is_valid()")
    return board


def _check_common(
    record: CandidateRecord | ProcessedRecord,
    *,
    seed: int,
    ratios: SplitRatios,
    opening_max_ply: int,
    endgame_material_max: int,
    check_split: bool,
) -> list[ValidationIssue]:
    """Shared candidate/processed field checks."""

    issues: list[ValidationIssue] = []

    try:
        board = _load_board(record.fen)
    except (ValueError, TypeError) as exc:
        issues.append(
            _issue("invalid_fen", f"Invalid FEN: {exc}", record)
        )
        return issues

    if board.ply() != record.ply:
        issues.append(
            _issue(
                "ply_mismatch",
                (
                    f"Record ply={record.ply} but board "
                    f"ply={board.ply()}"
                ),
                record,
            )
        )

    try:
        move = chess.Move.from_uci(record.move_uci)
    except (ValueError, TypeError) as exc:
        issues.append(
            _issue(
                "invalid_move_uci",
                f"Cannot parse move_uci: {exc}",
                record,
            )
        )
        return issues

    if not board.is_legal(move):
        issues.append(
            _issue(
                "illegal_human_move",
                f"Move {record.move_uci} is illegal",
                record,
            )
        )

    expected_phase = classify_phase(
        board,
        board.ply(),
        opening_max_ply=opening_max_ply,
        endgame_material_max=endgame_material_max,
    )
    if expected_phase != record.phase:
        issues.append(
            _issue(
                "phase_mismatch",
                (
                    f"Record phase={record.phase.value}, "
                    f"expected={expected_phase.value}"
                ),
                record,
            )
        )

    if check_split:
        expected_split = assign_split(
            record.game_id,
            seed=seed,
            ratios=ratios,
        )
        if expected_split != record.split:
            issues.append(
                _issue(
                    "split_mismatch",
                    (
                        f"Record split={record.split.value}, "
                        f"expected={expected_split.value}"
                    ),
                    record,
                )
            )

    return issues


# ---------------------------------------------------------
# Row-level validators
# ---------------------------------------------------------

def validate_candidate_record(
    record: CandidateRecord,
    *,
    seed: int = DEFAULT_SEED,
    ratios: SplitRatios = DEFAULT_RATIOS,
    opening_max_ply: int = OPENING_MAX_PLY,
    endgame_material_max: int = ENDGAME_MATERIAL_MAX,
    check_split: bool = True,
) -> list[ValidationIssue]:
    """Validate one CandidateRecord. Returns zero or more issues."""

    if not isinstance(record, CandidateRecord):
        raise TypeError("record must be a CandidateRecord")

    return _check_common(
        record,
        seed=seed,
        ratios=ratios,
        opening_max_ply=opening_max_ply,
        endgame_material_max=endgame_material_max,
        check_split=check_split,
    )


def validate_processed_record(
    record: ProcessedRecord,
    *,
    seed: int = DEFAULT_SEED,
    ratios: SplitRatios = DEFAULT_RATIOS,
    opening_max_ply: int = OPENING_MAX_PLY,
    endgame_material_max: int = ENDGAME_MATERIAL_MAX,
    check_split: bool = True,
) -> list[ValidationIssue]:
    """Validate one ProcessedRecord including action encoding."""

    if not isinstance(record, ProcessedRecord):
        raise TypeError("record must be a ProcessedRecord")

    issues = _check_common(
        record,
        seed=seed,
        ratios=ratios,
        opening_max_ply=opening_max_ply,
        endgame_material_max=endgame_material_max,
        check_split=check_split,
    )

    # Cannot continue encoding checks without a usable board/move.
    if any(
        issue.code in {
            "invalid_fen",
            "invalid_move_uci",
            "illegal_human_move",
        }
        for issue in issues
    ):
        return issues

    board = chess.Board(record.fen)
    move = chess.Move.from_uci(record.move_uci)

    try:
        expected_hash = position_hash(board)
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "position_hash_error",
                f"Failed to compute position_hash: {exc}",
                record,
            )
        )
        return issues

    if expected_hash != record.position_hash:
        issues.append(
            _issue(
                "position_hash_mismatch",
                "Stored position_hash does not match encoding",
                record,
                position_hash_value=record.position_hash,
            )
        )

    try:
        expected_action = encode_move(board, move)
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "action_encode_error",
                f"Failed to encode move: {exc}",
                record,
                position_hash_value=record.position_hash,
            )
        )
        return issues

    if expected_action != record.action_id:
        issues.append(
            _issue(
                "action_id_mismatch",
                (
                    f"Stored action_id={record.action_id}, "
                    f"expected={expected_action}"
                ),
                record,
                position_hash_value=record.position_hash,
            )
        )

    try:
        decoded = decode_move(board, record.action_id)
        if decoded != move:
            issues.append(
                _issue(
                    "action_roundtrip_mismatch",
                    (
                        f"decode(encode) yielded {decoded.uci()}, "
                        f"expected {move.uci()}"
                    ),
                    record,
                    position_hash_value=record.position_hash,
                )
            )
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "action_decode_error",
                f"Failed to decode action_id: {exc}",
                record,
                position_hash_value=record.position_hash,
            )
        )

    try:
        expected_legal = tuple(legal_action_ids(board))
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "legal_actions_error",
                f"Failed to enumerate legal actions: {exc}",
                record,
                position_hash_value=record.position_hash,
            )
        )
        return issues

    if record.legal_action_ids != expected_legal:
        issues.append(
            _issue(
                "legal_action_ids_mismatch",
                "Stored legal_action_ids do not match board",
                record,
                position_hash_value=record.position_hash,
            )
        )

    if record.action_id not in record.legal_action_ids:
        issues.append(
            _issue(
                "target_not_in_legal",
                "action_id missing from legal_action_ids",
                record,
                position_hash_value=record.position_hash,
            )
        )

    if any(
        not 0 <= action_id < ACTION_SPACE_SIZE
        for action_id in record.legal_action_ids
    ):
        issues.append(
            _issue(
                "legal_action_out_of_range",
                "legal_action_ids contains out-of-range values",
                record,
                position_hash_value=record.position_hash,
            )
        )

    return issues


# ---------------------------------------------------------
# Dataset-level streaming validator
# ---------------------------------------------------------

class _DatasetIndex:
    """Temporary SQLite index for whole-dataset checks."""

    def __init__(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(
            prefix="it3160-validate-",
            suffix=".sqlite",
            delete=False,
        )
        self._tmp.close()
        self.path = Path(self._tmp.name)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=OFF")
        self._conn.execute("PRAGMA synchronous=OFF")
        self._conn.executescript(
            """
            CREATE TABLE game_splits (
                game_id TEXT NOT NULL,
                split TEXT NOT NULL,
                PRIMARY KEY (game_id, split)
            ) WITHOUT ROWID;

            CREATE TABLE game_phase (
                game_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                n INTEGER NOT NULL,
                PRIMARY KEY (game_id, phase)
            ) WITHOUT ROWID;

            CREATE TABLE positions (
                position_hash TEXT NOT NULL,
                split TEXT NOT NULL,
                n INTEGER NOT NULL,
                PRIMARY KEY (position_hash, split)
            ) WITHOUT ROWID;
            """
        )

    def observe(
        self,
        *,
        game_id: str,
        split: str,
        phase: str,
        position_hash_value: str | None,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO game_splits(game_id, split)
            VALUES (?, ?)
            """,
            (game_id, split),
        )
        self._conn.execute(
            """
            INSERT INTO game_phase(game_id, phase, n)
            VALUES (?, ?, 1)
            ON CONFLICT(game_id, phase)
            DO UPDATE SET n = game_phase.n + 1
            """,
            (game_id, phase),
        )

        if position_hash_value is not None:
            self._conn.execute(
                """
                INSERT INTO positions(position_hash, split, n)
                VALUES (?, ?, 1)
                ON CONFLICT(position_hash, split)
                DO UPDATE SET n = positions.n + 1
                """,
                (position_hash_value, split),
            )

    def commit(self) -> None:
        self._conn.commit()

    def cross_split_games(self) -> Iterator[tuple[str, int]]:
        rows = self._conn.execute(
            """
            SELECT game_id, COUNT(*) AS n_splits
            FROM game_splits
            GROUP BY game_id
            HAVING n_splits > 1
            """
        )
        for row in rows:
            yield row[0], row[1]

    def multi_phase_samples(self) -> Iterator[tuple[str, str, int]]:
        rows = self._conn.execute(
            """
            SELECT game_id, phase, n
            FROM game_phase
            WHERE n > 1
            """
        )
        for row in rows:
            yield row[0], row[1], row[2]

    def cross_split_positions(self) -> Iterator[tuple[str, int]]:
        rows = self._conn.execute(
            """
            SELECT position_hash, COUNT(*) AS n_splits
            FROM positions
            GROUP BY position_hash
            HAVING n_splits > 1
            """
        )
        for row in rows:
            yield row[0], row[1]

    def within_split_duplicates(self) -> int:
        row = self._conn.execute(
            """
            SELECT COALESCE(SUM(n - 1), 0)
            FROM positions
            WHERE n > 1
            """
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self._conn.close()
        self.path.unlink(missing_ok=True)


def validate_dataset(
    records: Iterable[CandidateRecord | ProcessedRecord],
    *,
    kind: str,
    quotas: dict[tuple[Split, Phase], int] | None = None,
    seed: int = DEFAULT_SEED,
    ratios: SplitRatios = DEFAULT_RATIOS,
    opening_max_ply: int = OPENING_MAX_PLY,
    endgame_material_max: int = ENDGAME_MATERIAL_MAX,
    check_split: bool = True,
    max_issues: int = 100,
    row_check: bool = True,
) -> ValidationReport:
    """
    Stream-validate an entire candidate or processed dataset.

    Args:
        records:
            Iterable of CandidateRecord or ProcessedRecord.
        kind:
            ``"candidate"`` or ``"processed"``.
        quotas:
            Optional expected counts keyed by (Split, Phase).
            Missing or short buckets are reported as errors.
        seed / ratios / phase thresholds:
            Used for per-row recomputation when row_check=True.
        max_issues:
            Cap stored issue samples.
        row_check:
            If False, only dataset-level aggregate checks run.
            Processed position hashes are trusted in this quick
            mode; candidate hashes are always computed from FEN.
    """

    if kind not in ("candidate", "processed"):
        raise ValueError(f"Unsupported kind: {kind}")

    if isinstance(max_issues, bool) or not isinstance(max_issues, int):
        raise TypeError("max_issues must be an integer")
    if max_issues < 0:
        raise ValueError("max_issues must be non-negative")
    if quotas is not None:
        for key, expected in quotas.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(key[0], Split)
                or not isinstance(key[1], Phase)
            ):
                raise TypeError("Quota keys must be (Split, Phase) pairs")
            if isinstance(expected, bool) or not isinstance(expected, int):
                raise TypeError("Quota counts must be integers")
            if expected < 0:
                raise ValueError("Quota counts must be non-negative")

    report = ValidationReport(
        kind=kind,
        max_issues=max_issues,
    )

    phase_counts: Counter[tuple[str, str]] = Counter()
    split_counts: Counter[str] = Counter()
    index = _DatasetIndex()
    progress = Progress(f"Validate {kind}", "records")

    try:
        for record in records:
            report.checked += 1
            progress.update(report.checked)

            if kind == "candidate" and not isinstance(
                record, CandidateRecord
            ):
                report.add_error(
                    ValidationIssue(
                        code="wrong_record_type",
                        message=(
                            "Expected CandidateRecord, got "
                            f"{type(record).__name__}"
                        ),
                    )
                )
                continue

            if kind == "processed" and not isinstance(
                record, ProcessedRecord
            ):
                report.add_error(
                    ValidationIssue(
                        code="wrong_record_type",
                        message=(
                            "Expected ProcessedRecord, got "
                            f"{type(record).__name__}"
                        ),
                    )
                )
                continue

            phase_counts[
                (record.split.value, record.phase.value)
            ] += 1
            split_counts[record.split.value] += 1

            position_hash_value: str | None = None
            if isinstance(record, ProcessedRecord) and not row_check:
                position_hash_value = record.position_hash
            else:
                try:
                    position_hash_value = position_hash(
                        _load_board(record.fen)
                    )
                except Exception:  # noqa: BLE001
                    position_hash_value = None

            index.observe(
                game_id=record.game_id,
                split=record.split.value,
                phase=record.phase.value,
                position_hash_value=position_hash_value,
            )

            if not row_check:
                continue

            if kind == "candidate":
                row_issues = validate_candidate_record(
                    record,  # type: ignore[arg-type]
                    seed=seed,
                    ratios=ratios,
                    opening_max_ply=opening_max_ply,
                    endgame_material_max=endgame_material_max,
                    check_split=check_split,
                )
            else:
                row_issues = validate_processed_record(
                    record,  # type: ignore[arg-type]
                    seed=seed,
                    ratios=ratios,
                    opening_max_ply=opening_max_ply,
                    endgame_material_max=endgame_material_max,
                    check_split=check_split,
                )

            for issue in row_issues:
                report.add_error(issue)

        log(f"Validate {kind}: checking leakage and quotas")
        index.commit()

        for game_id, n_splits in index.cross_split_games():
            report.add_error(
                ValidationIssue(
                    code="game_cross_split",
                    message=(
                        f"Game appears in {n_splits} splits"
                    ),
                    game_id=game_id,
                )
            )

        for game_id, phase, n in index.multi_phase_samples():
            report.add_error(
                ValidationIssue(
                    code="game_phase_oversample",
                    message=(
                        f"Game has {n} records for phase "
                        f"{phase}; max is 1"
                    ),
                    game_id=game_id,
                    phase=phase,
                )
            )

        for fingerprint, n_splits in index.cross_split_positions():
            report.add_error(
                ValidationIssue(
                    code="position_cross_split",
                    message=(
                        f"Position appears in {n_splits} splits"
                    ),
                    position_hash=fingerprint,
                )
            )

        within_split_dupes = index.within_split_duplicates()
        if within_split_dupes > 0:
            report.add_warning(
                ValidationIssue(
                    code="within_split_position_duplicates",
                    message=(
                        f"{within_split_dupes} extra occurrences "
                        "of positions repeated inside a split "
                        "(allowed by current dedup policy)"
                    ),
                )
            )

        quota_stats: dict[str, dict] = {}
        shortfalls: dict[str, int] = {}

        if quotas is not None:
            for (split, phase), expected in quotas.items():
                key = f"{split.value}/{phase.value}"
                actual = phase_counts[
                    (split.value, phase.value)
                ]
                quota_stats[key] = {
                    "expected": expected,
                    "actual": actual,
                }
                if actual < expected:
                    deficit = expected - actual
                    shortfalls[key] = deficit
                    report.add_error(
                        ValidationIssue(
                            code="quota_shortfall",
                            message=(
                                f"Quota shortfall for {key}: "
                                f"have {actual}, need {expected}"
                            ),
                            split=split.value,
                            phase=phase.value,
                        )
                    )
                elif actual > expected:
                    report.add_warning(
                        ValidationIssue(
                            code="quota_surplus",
                            message=(
                                f"Quota surplus for {key}: "
                                f"have {actual}, expected "
                                f"{expected}"
                            ),
                            split=split.value,
                            phase=phase.value,
                        )
                    )

        report.stats = {
            "split_counts": dict(split_counts),
            "phase_counts": {
                f"{split}/{phase}": count
                for (split, phase), count in sorted(
                    phase_counts.items()
                )
            },
            "within_split_position_duplicates": within_split_dupes,
            "quota": quota_stats,
            "quota_shortfalls": shortfalls,
        }

    finally:
        index.close()

    progress.finish(f"errors={report.error_count}, warnings={report.warning_count}")
    return report


def iter_and_validate(
    records: Iterable[CandidateRecord | ProcessedRecord],
    *,
    kind: str,
    **kwargs,
) -> tuple[list[CandidateRecord | ProcessedRecord], ValidationReport]:
    """
    Materialize an iterable while validating it.

    Intended for small pilot runs only. Prefer streaming
    validate_dataset() for production-scale data.
    """

    materialised = list(records)
    report = validate_dataset(
        materialised,
        kind=kind,
        **kwargs,
    )
    return materialised, report
