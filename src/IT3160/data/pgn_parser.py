""" 
Convert PGN files to pairs of (si, ai)
- Read PGN games from archives.
- Validate standard chess games.
- Generate stable game identifiers.
- Extract raw (state, move) records.
"""
import hashlib
import io
import json
import re
import zipfile

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, TypedDict

import chess
import chess.pgn


VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}
STANDARD_VARIANTS = {"standard", "chess"}

class RawPosition(TypedDict):
    """A state-action pair from a PGN game."""
    game_id: str
    source_month: str

    ply: int
    fen: str
    move_uci: str

    white_elo: int | None
    black_elo: int | None
    result: str


@dataclass(frozen=True, slots=True)
class ParsedGame:
    """A validated PGN game with source metadata."""

    game_id: str
    source_month: str
    game: chess.pgn.Game


@dataclass(slots=True)
class ParserStats:
    """Streaming audit counters, optionally shared across source archives.

    Result and termination histograms count every game seen, including
    rejected games. No rating or time-control filtering is applied.
    """

    games_seen: int = 0
    games_accepted: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    results: dict[str, int] = field(default_factory=dict)
    terminations: dict[str, int] = field(default_factory=dict)
    accepted_results: dict[str, int] = field(default_factory=dict)
    rejected_results: dict[str, int] = field(default_factory=dict)
    accepted_terminations: dict[str, int] = field(default_factory=dict)
    rejected_terminations: dict[str, int] = field(default_factory=dict)

    def record(self, game: chess.pgn.Game, reason: str | None) -> None:
        self.games_seen += 1
        result = game.headers.get("Result", "*")
        termination = game.headers.get("Termination", "unknown")
        self.results[result] = self.results.get(result, 0) + 1
        self.terminations[termination] = self.terminations.get(termination, 0) + 1
        if reason is None:
            self.games_accepted += 1
            result_counts = self.accepted_results
            termination_counts = self.accepted_terminations
        else:
            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
            result_counts = self.rejected_results
            termination_counts = self.rejected_terminations
        result_counts[result] = result_counts.get(result, 0) + 1
        termination_counts[termination] = termination_counts.get(termination, 0) + 1

    def to_dict(self) -> dict:
        return {
            "games_seen": self.games_seen,
            "games_accepted": self.games_accepted,
            "games_rejected": self.games_seen - self.games_accepted,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "results": dict(sorted(self.results.items())),
            "terminations": dict(sorted(self.terminations.items())),
            "accepted_results": dict(sorted(self.accepted_results.items())),
            "rejected_results": dict(sorted(self.rejected_results.items())),
            "accepted_terminations": dict(sorted(self.accepted_terminations.items())),
            "rejected_terminations": dict(sorted(self.rejected_terminations.items())),
        }


class _CompletedGameBuilder(chess.pgn.GameBuilder):
    """Reject absent, conflicting, or nonterminal movetext results.

    python-chess deliberately accepts incomplete PGN and trusts a Result
    header even when it conflicts with the movetext. Dataset ingestion must
    distinguish those partial games from completed games.
    """

    def begin_game(self) -> None:
        super().begin_game()
        self._header_result: str | None = None
        self._movetext_result: str | None = None

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        super().visit_header(tagname, tagvalue)
        if tagname == "Result":
            self._header_result = tagvalue

    def visit_result(self, result: str) -> None:
        if self._movetext_result is not None:
            self.handle_error(ValueError("Multiple movetext result markers"))
        if self._header_result is not None and result != self._header_result:
            self.handle_error(ValueError("Header and movetext results disagree"))
        self._movetext_result = result
        super().visit_result(result)

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        if len(self.variation_stack) == 1 and self._movetext_result is not None:
            self.handle_error(ValueError("Mainline move follows the game result"))
        super().visit_move(board, move)

    def end_game(self) -> None:
        if self._movetext_result is None:
            self.handle_error(ValueError("Missing movetext result marker"))
        super().end_game()

    def handle_error(self, error: Exception) -> None:
        # Audit counters report rejected games without logging every bad game.
        self.game.errors.append(error)


class _CheckedPGNStream:
    """Validate ignored syntax while retaining line-by-line ZIP streaming.

    The chess parser ignores unmatched text, malformed headers and incomplete
    comments/variations. This adapter records those errors without loading a
    game or archive into memory. SAN interpretation stays with python-chess.
    """

    _GAP = re.compile(r"(?:\s|\d+\.(?:\.\.)?|\.{3}|[+#])+")

    def __init__(self, stream: io.TextIOWrapper):
        self.stream = stream
        self.begin_game()

    def begin_game(self) -> None:
        self.errors: list[Exception] = []
        self.in_comment = False
        self.variation_depth = 0

    def readline(self) -> str:
        line = self.stream.readline()
        self._check_line(line)
        return line

    def _check_line(self, line: str) -> None:
        if not self.in_comment and line.startswith(("%", ";")):
            return
        if not self.in_comment and line.lstrip("\ufeff").startswith("["):
            if chess.pgn.TAG_REGEX.fullmatch(line.lstrip("\ufeff")) is None:
                self.errors.append(ValueError("Malformed PGN header"))
            return

        offset = 0
        while offset < len(line):
            if self.in_comment:
                close = line.find("}", offset)
                if close == -1:
                    return
                self.in_comment = False
                offset = close + 1
                continue

            if line[offset] == "{":
                self.in_comment = True
                offset += 1
                continue
            if line[offset] == ";":
                return

            gap = self._GAP.match(line, offset)
            if gap is not None:
                offset = gap.end()
                continue
            token = chess.pgn.MOVETEXT_REGEX.match(line, offset)
            if token is None:
                self.errors.append(ValueError("Unrecognized PGN movetext"))
                return

            value = token.group(0)
            if value == "(":
                self.variation_depth += 1
            elif value == ")":
                if self.variation_depth == 0:
                    self.errors.append(ValueError("Unmatched PGN variation end"))
                else:
                    self.variation_depth -= 1
            offset = token.end()

    def finish_game(self, game: chess.pgn.Game) -> None:
        game.errors.extend(self.errors)
        if self.in_comment:
            game.errors.append(ValueError("Unterminated PGN comment"))
        if self.variation_depth:
            game.errors.append(ValueError("Unterminated PGN variation"))


def _parse_elo(value: str | None) -> int | None:
    """Convert a PGN Elo header to an integer, if available."""
    if value is None:
        return None

    try:
        elo = int(value)
        return elo if elo >= 0 else None
    except (TypeError, ValueError):
        return None


def is_valid_game(game: chess.pgn.Game) -> bool:
    """
    Check whether a PGN game can be used for training.

    Requirements:
    - No PGN parsing errors.
    - Standard chess from the normal initial position.
    - A known game result.
    - At least one move.
    - Every move can be legally replayed.

    PGN text completeness is checked by iter_valid_games(): a standalone
    chess.pgn.Game does not retain ignored raw text or its result marker.
    """

    return _game_rejection_reason(game) is None


def _game_rejection_reason(game: chess.pgn.Game) -> str | None:
    """Return one stable reason per rejected game for source audit reports."""

    if game.errors:
        return "pgn_parse_error"

    variant = game.headers.get("Variant", "Standard")
    if variant.strip().lower() not in STANDARD_VARIANTS:
        return "nonstandard_variant"

    if game.headers.get("Result", "*") not in VALID_RESULTS:
        return "unknown_result"

    try:
        board = game.board()
    except (ValueError, TypeError):
        return "invalid_initial_board"

    if board.chess960:
        return "chess960"

    if board.fen() != chess.STARTING_FEN:
        return "nonstandard_initial_position"

    num_moves = 0

    for move in game.mainline_moves():
        if not board.is_legal(move):
            return "illegal_mainline_move"

        board.push(move)
        num_moves += 1

    return None if num_moves > 0 else "no_moves"


def make_game_id(game: chess.pgn.Game) -> str:
    """
    Generate a deterministic game ID.
    The ID depends on game metadata and the mainline moves.
    """

    headers = game.headers

    identity = {
        "site": headers.get("Site", "").strip(),
        "date": headers.get("Date", "").strip(),
        "utc_date": headers.get("UTCDate", "").strip(),
        "utc_time": headers.get("UTCTime", "").strip(),
        "white": headers.get("White", "").strip(),
        "black": headers.get("Black", "").strip(),
        "result": headers.get("Result", "").strip(),
        "moves": [
            move.uci()
            for move in game.mainline_moves()
        ],
    }

    payload = json.dumps(
        identity,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.blake2b(
        payload,
        digest_size=16,
    ).hexdigest()


def iter_valid_games(
    archive_path: str | Path,
    source_month: str,
    max_games: int | None = None,
    *,
    stats: ParserStats | None = None,
) -> Iterator[ParsedGame]:
    """
    Stream valid games from a ZIP archive containing PGN files.

    Args:
        archive_path:
            Path to a local ZIP archive.
        source_month:
            Source month in YYYY-MM format.
        max_games:
            Maximum number of VALID games to yield.
            None means no limit.
        stats:
            Optional mutable audit counters. Counts only games read before
            max_games valid games have been yielded.

    Yields:
        ParsedGame objects, one at a time.
    """

    if not isinstance(source_month, str) or not re.fullmatch(
        r"[0-9]{4}-(0[1-9]|1[0-2])",
        source_month,
    ):
        raise ValueError("source_month must use YYYY-MM format")

    if max_games is not None:
        if isinstance(max_games, bool) or not isinstance(max_games, int):
            raise TypeError("max_games must be an integer or None")
        if max_games < 0:
            raise ValueError("max_games must be non-negative")

    if stats is not None and not isinstance(stats, ParserStats):
        raise TypeError("stats must be a ParserStats or None")

    if max_games == 0:
        return

    archive_path = Path(archive_path)
    accepted = 0

    with zipfile.ZipFile(archive_path, "r") as archive:
        pgn_files = sorted(
            (
                info
                for info in archive.infolist()
                if not info.is_dir()
                and info.filename.lower().endswith(".pgn")
            ),
            key=lambda info: info.filename,
        )

        if not pgn_files:
            raise ValueError(
                f"No PGN files found in {archive_path}"
            )

        for pgn_info in pgn_files:
            with archive.open(pgn_info, "r") as binary:
                with io.TextIOWrapper(
                    binary,
                    encoding="utf-8-sig",
                    errors="replace",
                ) as stream:
                    checked_stream = _CheckedPGNStream(stream)

                    while True:
                        checked_stream.begin_game()
                        game = chess.pgn.read_game(
                            checked_stream,
                            Visitor=_CompletedGameBuilder,
                        )

                        if game is None:
                            break

                        checked_stream.finish_game(game)
                        reason = _game_rejection_reason(game)
                        if stats is not None:
                            stats.record(game, reason)

                        if reason is not None:
                            continue

                        yield ParsedGame(
                            game_id=make_game_id(game),
                            source_month=source_month,
                            game=game,
                        )

                        accepted += 1

                        if (
                            max_games is not None
                            and accepted >= max_games
                        ):
                            return


def extract_positions(
    parsed_game: ParsedGame,
) -> Iterator[RawPosition]:
    """
    Extract raw positions from the mainline of a parsed game.
    Each record represents the board BEFORE the human move.
    """

    game = parsed_game.game
    board = game.board()

    white_elo = _parse_elo(
        game.headers.get("WhiteElo")
    )
    black_elo = _parse_elo(
        game.headers.get("BlackElo")
    )
    result = game.headers["Result"]

    for move in game.mainline_moves():
        if not board.is_legal(move):
            raise ValueError(
                f"Illegal move in game "
                f"{parsed_game.game_id}: {move.uci()}"
            )

        yield RawPosition(
            game_id=parsed_game.game_id,
            source_month=parsed_game.source_month,
            ply=board.ply(),
            fen=board.fen(),
            move_uci=move.uci(),
            white_elo=white_elo,
            black_elo=black_elo,
            result=result,
        )

        board.push(move)
