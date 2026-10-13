"""
End-to-end data pipeline orchestration.

Stages:
1. Parse / split / phase-sample → interim candidate shards
2. Index the full candidate pool and finalize deduplication
3. Filter safe positions, encode, fill quotas → processed shards
4. Validate and write reproducibility manifests
5. Optionally upload artifacts to Hugging Face

Candidate refill is supported by appending sources, then
rebuilding the dedup index over the *entire* candidate pool
before re-emitting processed data. Never borrow samples
across splits to fill a shortfall.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

import chess
import yaml

from .action_encoding import encode_move, legal_action_ids
from .deduplication import (
    GAME_HASH_VERSION,
    POSITION_HASH_VERSION,
    DeduplicationIndex,
    game_content_hash,
    position_hash,
)
from .pgn_parser import ParserStats, iter_valid_games
from .progress import Progress, log
from .sampling import _sample_game_candidates
from .schema import (
    CandidateRecord,
    Phase,
    ProcessedRecord,
    Split,
)
from .split import (
    DEFAULT_SEED,
    SPLIT_HASH_VERSION,
    SplitRatios,
    assign_split,
)
from .storage import (
    REPO_ID,
    SCHEMA_VERSION,
    make_shard_path,
    read_manifest,
    read_shard,
    upload_files_atomic,
    compute_checksum,
    verify_checksum,
    write_manifest,
    write_shard,
)
from .validation import ValidationReport, validate_dataset


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PhaseConfig:
    opening_max_ply: int = 24
    endgame_material_max: int = 8

    def __post_init__(self):
        for name in ("opening_max_ply", "endgame_material_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")


@dataclass(frozen=True, slots=True)
class QuotaConfig:
    train_per_phase: int = 450_000
    validation_per_phase: int = 25_000
    test_per_phase: int = 25_000

    def __post_init__(self):
        for name in (
            "train_per_phase",
            "validation_per_phase",
            "test_per_phase",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be a non-negative int"
                )

    def expected(self) -> dict[tuple[Split, Phase], int]:
        per_split = {
            Split.TRAIN: self.train_per_phase,
            Split.VALIDATION: self.validation_per_phase,
            Split.TEST: self.test_per_phase,
        }
        return {
            (split, phase): count
            for split, count in per_split.items()
            for phase in Phase
        }

    def for_split(self, split: Split) -> int:
        if split is Split.TRAIN:
            return self.train_per_phase
        if split is Split.VALIDATION:
            return self.validation_per_phase
        return self.test_per_phase


@dataclass(frozen=True, slots=True)
class DataConfig:
    seed: int = DEFAULT_SEED
    split: SplitRatios = field(default_factory=SplitRatios)
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    max_samples_per_game_per_phase: int = 1
    shard_size: int = 50_000

    def __post_init__(self):
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer")

        if (type(self.max_samples_per_game_per_phase) is not int
                or self.max_samples_per_game_per_phase != 1):
            raise ValueError(
                "Only max_samples_per_game_per_phase=1 "
                "is supported"
            )

        if (
            isinstance(self.shard_size, bool)
            or not isinstance(self.shard_size, int)
            or self.shard_size <= 0
        ):
            raise ValueError("shard_size must be a positive int")
        if not isinstance(self.split, SplitRatios):
            raise TypeError("split must be SplitRatios")
        if not isinstance(self.phase, PhaseConfig):
            raise TypeError("phase must be PhaseConfig")
        if not isinstance(self.quota, QuotaConfig):
            raise TypeError("quota must be QuotaConfig")

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "split": {
                "train": self.split.train,
                "validation": self.split.validation,
                "test": self.split.test,
            },
            "phase": {
                "opening_max_ply": self.phase.opening_max_ply,
                "endgame_material_max": (
                    self.phase.endgame_material_max
                ),
            },
            "quota": {
                "train_per_phase": self.quota.train_per_phase,
                "validation_per_phase": (
                    self.quota.validation_per_phase
                ),
                "test_per_phase": self.quota.test_per_phase,
            },
            "max_samples_per_game_per_phase": (
                self.max_samples_per_game_per_phase
            ),
            "shard_size": self.shard_size,
        }


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One monthly raw ZIP to ingest."""

    archive_path: Path
    source_month: str
    max_games: int | None = None
    hf_repo_id: str | None = None
    hf_path: str | None = None
    hf_revision: str | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "archive_path",
            Path(self.archive_path),
        )

        if not isinstance(self.source_month, str) or not re.fullmatch(
            r"[0-9]{4}-(0[1-9]|1[0-2])",
            self.source_month,
        ):
            raise ValueError(
                "source_month must use YYYY-MM format"
            )
        if self.max_games is not None and (
            type(self.max_games) is not int or self.max_games < 0
        ):
            raise ValueError("max_games must be a non-negative int")
        origin = (self.hf_repo_id, self.hf_path, self.hf_revision)
        if any(value is not None for value in origin):
            if not all(isinstance(value, str) and value.strip() for value in origin):
                raise ValueError("Hugging Face source metadata must include repo, path and revision")
            path = PurePosixPath(self.hf_path)
            if path.is_absolute() or ".." in path.parts or "\\" in self.hf_path:
                raise ValueError("Invalid Hugging Face source path")
            if not re.fullmatch(r"[0-9a-fA-F]{40,64}", self.hf_revision):
                raise ValueError("Hugging Face source revision must be a pinned commit SHA")


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    """Local working directories for one run_id."""

    work_dir: Path
    run_id: str

    def __post_init__(self):
        object.__setattr__(self, "work_dir", Path(self.work_dir))

        if not isinstance(self.run_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*",
            self.run_id,
        ):
            raise ValueError("Invalid run_id")

    @property
    def root(self) -> Path:
        return self.work_dir / self.run_id

    @property
    def candidates_dir(self) -> Path:
        return self.root / "interim" / "candidates"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def dedup_db(self) -> Path:
        return self.root / "dedup" / "index.sqlite"

    @property
    def accepted_games_path(self) -> Path:
        return self.root / "dedup" / "accepted_games.json"

    @property
    def sources_manifest_path(self) -> Path:
        return self.manifests_dir / "sources.json"

    @property
    def run_manifest_path(self) -> Path:
        return self.manifests_dir / "run.json"

    @property
    def validation_manifest_path(self) -> Path:
        return self.manifests_dir / "validation.json"

    def ensure(self) -> None:
        for path in (
            self.candidates_dir,
            self.processed_dir,
            self.manifests_dir,
            self.dedup_db.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


class QuotaShortfallError(RuntimeError):
    """Raised when processed quotas cannot be filled."""

    def __init__(self, shortfalls: dict[str, int]):
        self.shortfalls = dict(shortfalls)
        detail = ", ".join(
            f"{key}: missing {count}"
            for key, count in sorted(shortfalls.items())
        )
        super().__init__(
            f"Quota shortfall after deduplication: {detail}"
        )


# ---------------------------------------------------------
# Config loading
# ---------------------------------------------------------

def load_data_config(
    path: str | Path | None = None,
) -> DataConfig:
    """Load DataConfig from YAML, or return defaults."""

    if path is None:
        return DataConfig()

    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    unknown = set(raw) - set(DataConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    values = dict(raw)
    for name, cls in (("split", SplitRatios), ("phase", PhaseConfig), ("quota", QuotaConfig)):
        section = raw.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"Config {name} must be a mapping")
        unknown = set(section) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
        values[name] = cls(**section)
    # Preserve YAML types: coercion used to silently turn 2.5 into 2 and True into 1.
    return DataConfig(**values)


def normalize_sources(
    sources: Iterable[SourceSpec],
) -> list[SourceSpec]:
    """Sort sources by month then archive path for stability."""

    ordered = sorted(
        sources,
        key=lambda source: (
            source.source_month,
            str(source.archive_path),
        ),
    )

    months = [source.source_month for source in ordered]
    if len(months) != len(set(months)):
        raise ValueError(
            "Duplicate source_month values in sources"
        )

    return ordered


def _check_refill(config: DataConfig, paths: PipelinePaths, ordered: list[SourceSpec]) -> dict:
    if not paths.sources_manifest_path.is_file():
        raise ValueError("Cannot append without an existing sources manifest")
    previous = read_manifest(paths.sources_manifest_path)
    if previous.get("config") != config.to_dict():
        raise ValueError("Refill must use the original data config; start a new run for changes")
    old_months = {source["source_month"] for source in previous.get("sources", [])}
    if old_months & {source.source_month for source in ordered}:
        raise ValueError("Refill requires new source months")
    verify_shard_inventory(paths.candidates_dir, previous.get("candidate_shards", []))
    return previous


# ---------------------------------------------------------
# Encoding
# ---------------------------------------------------------

def encode_candidate(
    candidate: CandidateRecord,
) -> ProcessedRecord:
    """Convert a CandidateRecord into a ProcessedRecord."""

    board = chess.Board(candidate.fen)
    move = chess.Move.from_uci(candidate.move_uci)

    return ProcessedRecord(
        game_id=candidate.game_id,
        source_month=candidate.source_month,
        split=candidate.split,
        phase=candidate.phase,
        ply=candidate.ply,
        fen=candidate.fen,
        move_uci=candidate.move_uci,
        position_hash=position_hash(board),
        action_id=encode_move(board, move),
        legal_action_ids=tuple(legal_action_ids(board)),
        white_elo=candidate.white_elo,
        black_elo=candidate.black_elo,
        result=candidate.result,
    )


# ---------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------

class _ShardWriter:
    """Buffer records and flush Parquet shards of fixed size."""

    def __init__(
        self,
        *,
        directory: Path,
        kind: str,
        shard_size: int,
        stage: str,
        run_id: str,
        split: Split,
        start_index: int = 0,
    ):
        self.directory = directory
        self.kind = kind
        self.shard_size = shard_size
        self.stage = stage
        self.run_id = run_id
        self.split = split
        self.next_index = start_index
        self.buffer: list = []
        self.shard_infos: list[dict] = []
        self.total_rows = 0

    def add(self, record) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        remote_name = Path(
            make_shard_path(
                self.stage,  # type: ignore[arg-type]
                self.run_id,
                self.split,
                self.next_index,
            )
        ).name

        local_path = (
            self.directory
            / self.split.value
            / remote_name
        )

        info = write_shard(
            self.buffer,
            local_path,
            kind=self.kind,  # type: ignore[arg-type]
            overwrite=True,
        )

        self.shard_infos.append(
            {
                "local_path": str(local_path),
                "remote_path": make_shard_path(
                    self.stage,  # type: ignore[arg-type]
                    self.run_id,
                    self.split,
                    self.next_index,
                ),
                "rows": info.rows,
                "sha256": info.sha256,
                "size_bytes": info.size_bytes,
                "phase_counts": info.phase_counts,
                "split_counts": info.split_counts,
            }
        )

        self.total_rows += info.rows
        self.next_index += 1
        self.buffer.clear()

    def close(self) -> list[dict]:
        self.flush()
        return self.shard_infos


def _next_shard_index(directory: Path, split: Split) -> int:
    split_dir = directory / split.value
    if not split_dir.exists():
        return 0

    indices = []
    for path in split_dir.glob("part-*.parquet"):
        match = re.fullmatch(
            r"part-(\d{5})\.parquet",
            path.name,
        )
        if match:
            indices.append(int(match.group(1)))

    return (max(indices) + 1) if indices else 0


def iter_local_shards(
    directory: Path,
    *,
    kind: str,
    splits: Iterable[Split] | None = None,
) -> Iterator:
    """Yield records from local Parquet shards in stable order."""

    split_list = (
        list(splits) if splits is not None else list(Split)
    )

    for split in split_list:
        split_dir = directory / split.value
        if not split_dir.exists():
            continue

        for path in sorted(split_dir.glob("part-*.parquet")):
            yield from read_shard(
                path,
                kind=kind,  # type: ignore[arg-type]
            )


def list_local_shard_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for split in Split:
        split_dir = directory / split.value
        if split_dir.exists():
            files.extend(
                sorted(split_dir.glob("part-*.parquet"))
            )
    return files


def verify_shard_inventory(directory: Path, shard_infos: list[dict]) -> None:
    """Verify the exact file set and checksums, using portable split/name paths."""
    expected: dict[Path, dict] = {}
    for info in shard_infos:
        remote = Path(info["remote_path"])
        if remote.parent.name not in {split.value for split in Split}:
            raise ValueError("Invalid split in shard inventory")
        local = directory / remote.parent.name / remote.name
        if local in expected:
            raise ValueError("Duplicate shard in inventory")
        expected[local] = info
    actual = set(list_local_shard_files(directory))
    if actual != set(expected):
        raise ValueError("Shard inventory mismatch: missing or unexpected files")
    for local, info in expected.items():
        verify_checksum(local, info["sha256"])


def _clear_split_shards(directory: Path) -> None:
    for split in Split:
        split_dir = directory / split.value
        if not split_dir.exists():
            continue
        for stale in split_dir.glob("part-*.parquet"):
            stale.unlink()


# ---------------------------------------------------------
# Stage 1: candidate pool
# ---------------------------------------------------------

@dataclass(slots=True)
class CandidateBuildResult:
    sources: list[dict]
    shard_infos: list[dict]
    candidate_counts: dict[str, int]
    games_seen: int
    games_sampled: int


def build_candidate_pool(
    sources: Iterable[SourceSpec],
    config: DataConfig,
    paths: PipelinePaths,
    *,
    append: bool = False,
) -> CandidateBuildResult:
    """
    Parse sources, sample per-game candidates, write shards.

    If ``append`` is True, new shards continue after existing
    part files. Callers that append must rebuild the dedup
    index over the entire candidate pool afterwards.
    """

    ordered = normalize_sources(sources)
    if not ordered:
        raise ValueError("At least one source is required")
    for source in ordered:
        if not source.archive_path.is_file():
            raise FileNotFoundError(source.archive_path)

    previous_sources: list[dict] = []
    previous_shards: list[dict] = []
    if append:
        previous = _check_refill(config, paths, ordered)
        previous_sources = list(previous.get("sources", []))
        previous_shards = list(previous.get("candidate_shards", []))

    paths.ensure()

    if not append:
        _clear_split_shards(paths.candidates_dir)

    files_before = set(list_local_shard_files(paths.candidates_dir))
    try:
        writers = {
            split: _ShardWriter(
                directory=paths.candidates_dir,
                kind="candidate",
                shard_size=config.shard_size,
                stage="interim",
                run_id=paths.run_id,
                split=split,
                start_index=(
                    _next_shard_index(
                        paths.candidates_dir,
                        split,
                    )
                    if append
                    else 0
                ),
            )
            for split in Split
        }

        games_seen = 0
        games_sampled = 0
        candidate_counts: Counter[str] = Counter()
        source_summaries: list[dict] = []
        seen_content: set[str] = set()

        if append:
            for existing in iter_local_shards(
                paths.candidates_dir,
                kind="candidate",
            ):
                seen_content.add(existing.game_id)

        for source in ordered:
            source_games = 0
            source_candidates = 0
            parser_stats = ParserStats()
            progress = Progress(f"Build candidates [{source.source_month}]", "valid games")

            for parsed in iter_valid_games(
                source.archive_path,
                source.source_month,
                max_games=source.max_games,
                stats=parser_stats,
            ):
                games_seen += 1
                source_games += 1
                progress.update(source_games)

                content_id = game_content_hash(parsed.game)
                if content_id in seen_content:
                    continue

                seen_content.add(content_id)

                split = assign_split(
                    content_id,
                    seed=config.seed,
                    ratios=config.split,
                )

                # iter_valid_games has just validated every mainline move.
                # Do not persist a "validated" flag on the mutable Game object.
                candidates = _sample_game_candidates(
                    parsed,
                    split,
                    content_id=content_id,
                    check_legal_moves=False,
                    seed=config.seed,
                    opening_max_ply=(
                        config.phase.opening_max_ply
                    ),
                    endgame_material_max=(
                        config.phase.endgame_material_max
                    ),
                    max_samples_per_phase=(
                        config.max_samples_per_game_per_phase
                    ),
                )

                if not candidates:
                    continue

                games_sampled += 1

                for candidate in candidates:
                    writers[candidate.split].add(candidate)
                    key = (
                        f"{candidate.split.value}/"
                        f"{candidate.phase.value}"
                    )
                    candidate_counts[key] += 1
                    source_candidates += 1

            progress.finish(
                f"{source_candidates:,} candidates, "
                f"{parser_stats.games_seen - parser_stats.games_accepted:,} rejected games"
            )
            source_summaries.append(
                {
                    "archive_path": str(source.archive_path.resolve()),
                    "source_month": source.source_month,
                    "max_games": source.max_games,
                    "sha256": compute_checksum(source.archive_path),
                    "size_bytes": source.archive_path.stat().st_size,
                    "games_accepted": source_games,
                    "candidates_emitted": source_candidates,
                    "parser": parser_stats.to_dict(),
                    **({"hf_repo_id": source.hf_repo_id, "hf_path": source.hf_path,
                        "hf_revision": source.hf_revision} if source.hf_repo_id else {}),
                }
            )

        shard_infos: list[dict] = []
        for writer in writers.values():
            shard_infos.extend(writer.close())

        write_manifest(
            {
                "run_id": paths.run_id,
                "append": append,
                "config": config.to_dict(),
                "sources": previous_sources + source_summaries,
                "candidate_shards": previous_shards + shard_infos,
                "identity_policy": {
                    "game_id": "game_content_hash",
                    "game_hash_version": GAME_HASH_VERSION,
                    "note": (
                        "Two games with identical initial FEN and "
                        "mainline UCI moves share one game_id even "
                        "if PGN metadata differs."
                    ),
                },
            },
            paths.sources_manifest_path,
            overwrite=True,
        )

        return CandidateBuildResult(
            sources=source_summaries,
            shard_infos=shard_infos,
            candidate_counts=dict(candidate_counts),
            games_seen=games_seen,
            games_sampled=games_sampled,
        )

    except BaseException:
        if append:
            for new_file in set(list_local_shard_files(paths.candidates_dir)) - files_before:
                new_file.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------
# Stage 2: deduplication index
# ---------------------------------------------------------

@dataclass(slots=True)
class DedupBuildResult:
    stats: dict
    accepted_games: int
    indexed_candidates: int


def build_dedup_index(
    paths: PipelinePaths,
) -> DedupBuildResult:
    """
    Scan the full local candidate pool and finalize a new index.

    Always rebuilds from scratch so candidate refill remains safe.
    """

    paths.ensure()
    if paths.sources_manifest_path.exists():
        source_manifest = read_manifest(paths.sources_manifest_path)
        verify_shard_inventory(paths.candidates_dir, source_manifest["candidate_shards"])

    if paths.dedup_db.exists():
        paths.dedup_db.unlink()
    if paths.accepted_games_path.exists():
        paths.accepted_games_path.unlink()

    accepted_set: set[str] = set()
    rejected_duplicate_games = 0
    indexed_candidates = 0
    progress = Progress("Build dedup index", "candidates")

    with DeduplicationIndex(paths.dedup_db) as index:
        for candidate in iter_local_shards(
            paths.candidates_dir,
            kind="candidate",
        ):
            if candidate.game_id not in accepted_set:
                is_new = index.register_game(candidate.game_id)
                if not is_new:
                    rejected_duplicate_games += 1
                    continue
                accepted_set.add(candidate.game_id)

            board = chess.Board(candidate.fen)
            fingerprint = position_hash(board)
            index.register_position(
                fingerprint,
                candidate.split,
            )
            indexed_candidates += 1
            progress.update(indexed_candidates)

        log("Build dedup index: finalizing")
        index.finalize()
        stats = dict(index.stats())

    stats["rejected_duplicate_game_rows"] = (
        rejected_duplicate_games
    )

    write_manifest(
        {
            "accepted_games": sorted(accepted_set),
            "count": len(accepted_set),
        },
        paths.accepted_games_path,
        overwrite=True,
    )

    progress.finish(f"{len(accepted_set):,} unique games")
    return DedupBuildResult(
        stats=stats,
        accepted_games=len(accepted_set),
        indexed_candidates=indexed_candidates,
    )


# ---------------------------------------------------------
# Stage 3: filter, encode, quota
# ---------------------------------------------------------

SELECTION_VERSION = "hash-priority-shuffle-v1"


def _priority(seed: int, purpose: str, identity: str) -> str:
    payload = json.dumps([SELECTION_VERSION, seed, purpose, identity], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ProcessedBuildResult:
    shard_infos: list[dict]
    selected_counts: dict[str, int]
    shortfalls: dict[str, int]
    skipped_cross_split: int
    skipped_within_split_duplicate: int
    skipped_unaccepted_game: int
    complete: bool


def build_processed_dataset(
    config: DataConfig,
    paths: PipelinePaths,
    *,
    require_full_quota: bool = True,
) -> ProcessedBuildResult:
    """
    Filter safe candidates, encode, fill quotas, write shards.

    Use seeded hash priorities to sample quotas across the entire safe pool.
    Deduplicate positions before selecting and independently shuffle each split.
    SQLite stores the selection pool on disk instead of retaining candidate rows in RAM.
    """

    paths.ensure()

    if not paths.dedup_db.exists():
        raise FileNotFoundError(
            f"Missing dedup index: {paths.dedup_db}"
        )
    if not paths.accepted_games_path.exists():
        raise FileNotFoundError(
            f"Missing accepted games list: "
            f"{paths.accepted_games_path}"
        )

    accepted_payload = read_manifest(paths.accepted_games_path)
    accepted_set = set(accepted_payload.get("accepted_games", []))

    _clear_split_shards(paths.processed_dir)

    quotas = config.quota.expected()
    filled: Counter[tuple[Split, Phase]] = Counter()
    selected_counts: Counter[str] = Counter()

    writers = {
        split: _ShardWriter(
            directory=paths.processed_dir,
            kind="processed",
            shard_size=config.shard_size,
            stage="processed",
            run_id=paths.run_id,
            split=split,
        )
        for split in Split
    }

    skipped_cross_split = 0
    skipped_within_split_duplicate = 0
    skipped_unaccepted_game = 0

    with tempfile.TemporaryDirectory(prefix="selection-", dir=paths.root) as temporary, \
            closing(sqlite3.connect(str(Path(temporary) / "selection.sqlite"))) as selection, \
            DeduplicationIndex(paths.dedup_db) as index:
        if index.stats()["status"] != "finalized":
            raise RuntimeError(
                "Dedup index must be finalized before filtering"
            )
        selection.executescript("""
            CREATE TABLE pool (
                fingerprint TEXT PRIMARY KEY, split TEXT NOT NULL, phase TEXT NOT NULL,
                representative TEXT NOT NULL, priority TEXT NOT NULL,
                shuffle TEXT NOT NULL, payload TEXT NOT NULL
            );
            CREATE TABLE selected (fingerprint TEXT PRIMARY KEY);
        """)

        filtering = Progress("Build processed: filter candidates", "candidates")
        scanned = 0
        for candidate in iter_local_shards(
            paths.candidates_dir,
            kind="candidate",
        ):
            scanned += 1
            filtering.update(scanned)
            if candidate.game_id not in accepted_set:
                skipped_unaccepted_game += 1
                continue

            board = chess.Board(candidate.fen)
            fingerprint = position_hash(board)

            if not index.is_safe_position(
                fingerprint,
                candidate.split,
            ):
                skipped_cross_split += 1
                continue

            identity = f"{candidate.game_id}/{candidate.ply}/{candidate.move_uci}"
            representative = _priority(config.seed, "representative", identity)
            # Keep a deterministic representative of repeated positions.
            cursor = selection.execute("""
                INSERT OR IGNORE INTO pool VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (fingerprint, candidate.split.value, candidate.phase.value, representative,
                  _priority(config.seed, "sample", fingerprint),
                  _priority(config.seed, "shuffle", fingerprint),
                  json.dumps(candidate.to_dict(), separators=(",", ":"))))
            if cursor.rowcount == 0:
                skipped_within_split_duplicate += 1
                selection.execute("""
                    UPDATE pool SET split=?, phase=?, representative=?, payload=?
                    WHERE fingerprint=? AND representative>?
                """, (candidate.split.value, candidate.phase.value, representative,
                      json.dumps(candidate.to_dict(), separators=(",", ":")),
                      fingerprint, representative))

        filtering.finish(f"{skipped_cross_split:,} cross-split conflicts skipped")
        log("Build processed: selecting quotas and shuffling")
        selection.execute("CREATE INDEX quota_order ON pool(split, phase, priority, fingerprint)")
        for (split, phase), target in quotas.items():
            selection.execute("""
                INSERT INTO selected SELECT fingerprint FROM pool
                WHERE split=? AND phase=? ORDER BY priority, fingerprint LIMIT ?
            """, (split.value, phase.value, target))
        selection.commit()

        encoding = Progress("Build processed: encode and write", "samples")
        encoded = 0
        for split in Split:
            log(f"Build processed: writing {split.value}")
            rows = selection.execute("""
                SELECT pool.payload FROM pool JOIN selected USING (fingerprint)
                WHERE pool.split=? ORDER BY pool.shuffle, pool.fingerprint
            """, (split.value,))
            for (payload,) in rows:
                values = json.loads(payload)
                values["split"] = Split(values["split"])
                values["phase"] = Phase(values["phase"])
                candidate = CandidateRecord(**values)
                processed = encode_candidate(candidate)
                writers[split].add(processed)
                filled[(split, candidate.phase)] += 1
                selected_counts[f"{split.value}/{candidate.phase.value}"] += 1
                encoded += 1
                encoding.update(encoded)

    shard_infos: list[dict] = []
    for writer in writers.values():
        shard_infos.extend(writer.close())

    shortfalls: dict[str, int] = {}
    for (split, phase), target in quotas.items():
        actual = filled[(split, phase)]
        if actual < target:
            shortfalls[f"{split.value}/{phase.value}"] = (
                target - actual
            )

    complete = len(shortfalls) == 0
    encoding.finish(f"{len(shard_infos):,} shards, {sum(shortfalls.values()):,} missing samples")

    if require_full_quota and not complete:
        raise QuotaShortfallError(shortfalls)

    return ProcessedBuildResult(
        shard_infos=shard_infos,
        selected_counts=dict(selected_counts),
        shortfalls=shortfalls,
        skipped_cross_split=skipped_cross_split,
        skipped_within_split_duplicate=(
            skipped_within_split_duplicate
        ),
        skipped_unaccepted_game=skipped_unaccepted_game,
        complete=complete,
    )


# ---------------------------------------------------------
# Stage 4: validation + manifests
# ---------------------------------------------------------

@dataclass(slots=True)
class PipelineResult:
    config: DataConfig
    paths: PipelinePaths
    candidate: CandidateBuildResult | None
    dedup: DedupBuildResult | None
    processed: ProcessedBuildResult | None
    validation: ValidationReport | None
    status: str
    manifest_path: Path | None = None


def validate_processed_shards(
    config: DataConfig,
    paths: PipelinePaths,
    *,
    row_check: bool = True,
    max_issues: int = 100,
) -> ValidationReport:
    """Validate local processed shards and write the report."""

    log(f"Validate: checking saved shards for {paths.run_id}")
    if paths.run_manifest_path.exists():
        previous = read_manifest(paths.run_manifest_path)
        if previous.get("config") != config.to_dict():
            raise ValueError("Validation config does not match the run manifest")
        if previous.get("processed") is not None:
            verify_shard_inventory(paths.processed_dir, previous["processed"]["shard_infos"])

    report = validate_dataset(
        iter_local_shards(
            paths.processed_dir,
            kind="processed",
        ),
        kind="processed",
        quotas=config.quota.expected(),
        seed=config.seed,
        ratios=config.split,
        opening_max_ply=config.phase.opening_max_ply,
        endgame_material_max=(
            config.phase.endgame_material_max
        ),
        check_split=True,
        max_issues=max_issues,
        row_check=row_check,
    )
    report.stats["row_check"] = row_check

    write_manifest(
        report.to_dict(),
        paths.validation_manifest_path,
        overwrite=True,
    )
    if paths.run_manifest_path.exists():
        previous["validation"] = report.to_dict()
        previous.setdefault("manifest_checksums", {})["validation.json"] = compute_checksum(
            paths.validation_manifest_path
        )
        if previous.get("status") != "building":
            previous["status"] = (
                "quota_shortfall" if not (previous.get("processed") or {}).get("complete")
                else "complete" if report.ok else "validation_failed"
            )
        write_manifest(previous, paths.run_manifest_path, overwrite=True)
    return report


def write_run_manifest(
    *,
    config: DataConfig,
    paths: PipelinePaths,
    candidate: CandidateBuildResult | None,
    dedup: DedupBuildResult | None,
    processed: ProcessedBuildResult | None,
    validation: ValidationReport | None,
    status: str,
) -> Path:
    """Write the top-level reproducibility manifest."""

    paths.ensure()
    dependencies = {}
    for name in ("chess", "numpy", "pyarrow", "PyYAML", "huggingface-hub"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = "unavailable"

    manifest_checksums = {
        path.name: compute_checksum(path)
        for path in (paths.sources_manifest_path, paths.validation_manifest_path)
        if path.is_file()
    }
    split_sha256 = None
    if paths.accepted_games_path.exists():
        accepted = read_manifest(paths.accepted_games_path)["accepted_games"]
        split_sha256 = hashlib.sha256(json.dumps([
            [game_id, assign_split(game_id, seed=config.seed, ratios=config.split).value]
            for game_id in sorted(accepted)
        ], separators=(",", ":")).encode()).hexdigest()

    payload = {
        "run_id": paths.run_id,
        "status": status,
        "config": config.to_dict(),
        "config_sha256": hashlib.sha256(json.dumps(
            config.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "manifest_checksums": manifest_checksums,
        "runtime": {"python": platform.python_version(), "packages": dependencies},
        "split_sha256": split_sha256,
        "versions": {
            "schema_version": SCHEMA_VERSION,
            "game_hash_version": GAME_HASH_VERSION,
            "position_hash_version": POSITION_HASH_VERSION,
            "selection_version": SELECTION_VERSION,
            "split_hash_version": SPLIT_HASH_VERSION,
            "sampling_version": "sampling-v1",
        },
        "identity_policy": {
            "game_id": "game_content_hash",
            "split_key": "game_content_hash",
            "note": (
                "Identical mainline move sequences share a "
                "game_id and therefore the same split, even "
                "when PGN metadata differs."
            ),
        },
        "dedup_policy": {
            "cross_split_positions": "rejected",
            "within_split_positions": (
                "lowest seeded hash priority per position"
            ),
            "duplicate_game_content": "first_wins",
        },
        "selection_policy": {
            "quota": "seeded hash priority over the entire safe candidate pool",
            "output_order": "independent seeded shuffle within each split",
        },
        "candidate": None
        if candidate is None
        else {
            "games_seen": candidate.games_seen,
            "games_sampled": candidate.games_sampled,
            "candidate_counts": candidate.candidate_counts,
            "shard_infos": candidate.shard_infos,
            "sources": candidate.sources,
        },
        "dedup": None
        if dedup is None
        else {
            "stats": dedup.stats,
            "accepted_games": dedup.accepted_games,
            "indexed_candidates": dedup.indexed_candidates,
        },
        "processed": None
        if processed is None
        else {
            "complete": processed.complete,
            "selected_counts": processed.selected_counts,
            "shortfalls": processed.shortfalls,
            "skipped_cross_split": processed.skipped_cross_split,
            "skipped_within_split_duplicate": (
                processed.skipped_within_split_duplicate
            ),
            "skipped_unaccepted_game": (
                processed.skipped_unaccepted_game
            ),
            "shard_infos": processed.shard_infos,
        },
        "validation": None
        if validation is None
        else validation.to_dict(),
    }

    return write_manifest(
        payload,
        paths.run_manifest_path,
        overwrite=True,
    )


# ---------------------------------------------------------
# Stage 5: optional HF upload
# ---------------------------------------------------------

def verify_run_artifacts(paths: PipelinePaths, *, require_complete: bool = False) -> dict:
    """Verify a saved run before reuse or publication; never trust status alone."""
    log(f"Verify artifacts: {paths.run_id}")
    manifest = read_manifest(paths.run_manifest_path)
    if manifest.get("run_id") != paths.run_id:
        raise ValueError("Run ID does not match its manifest")
    config_checksum = hashlib.sha256(json.dumps(
        manifest.get("config"), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if manifest.get("config_sha256") != config_checksum:
        raise ValueError("Config checksum mismatch")
    if require_complete:
        validation = manifest.get("validation") or {}
        if (manifest.get("status") != "complete" or not validation.get("ok")
                or not validation.get("stats", {}).get("row_check")):
            raise ValueError("Upload requires complete quotas and full row validation")
    checksums = manifest.get("manifest_checksums", {})
    required = {"sources.json"}
    if manifest.get("validation") is not None:
        required.add("validation.json")
    if not required <= checksums.keys():
        raise ValueError("Missing manifest checksums")
    for name, checksum in checksums.items():
        if name not in {"sources.json", "validation.json"}:
            raise ValueError("Unexpected manifest checksum path")
        verify_checksum(paths.manifests_dir / name, checksum)
    if paths.sources_manifest_path.exists():
        sources = read_manifest(paths.sources_manifest_path)
        verify_shard_inventory(paths.candidates_dir, sources["candidate_shards"])
    processed = manifest.get("processed")
    if processed is None:
        raise ValueError("Run has no processed dataset")
    verify_shard_inventory(paths.processed_dir, processed["shard_infos"])
    return manifest


def upload_run_artifacts(
    paths: PipelinePaths,
    *,
    include_candidates: bool = True,
    include_processed: bool = True,
    include_manifests: bool = True,
    repo_id: str = REPO_ID,
    token: str | None = None,
) -> list[str]:
    """Upload local run artifacts to the HF dataset repo."""

    progress = Progress(f"Upload [{paths.run_id}]", "files")
    verify_run_artifacts(paths, require_complete=True)

    commit_urls: list[str] = []
    files: dict[str, Path] = {}

    def _upload(local_path: Path, remote_path: str) -> None:
        files[remote_path] = local_path

    if include_candidates:
        for path in list_local_shard_files(paths.candidates_dir):
            split = Split(path.parent.name)
            match = re.fullmatch(
                r"part-(\d{5})\.parquet",
                path.name,
            )
            if not match:
                continue
            remote = make_shard_path(
                "interim",
                paths.run_id,
                split,
                int(match.group(1)),
            )
            _upload(path, remote)

    if include_processed:
        for path in list_local_shard_files(paths.processed_dir):
            split = Split(path.parent.name)
            match = re.fullmatch(
                r"part-(\d{5})\.parquet",
                path.name,
            )
            if not match:
                continue
            remote = make_shard_path(
                "processed",
                paths.run_id,
                split,
                int(match.group(1)),
            )
            _upload(path, remote)

    if include_manifests:
        for path in sorted(paths.manifests_dir.glob("*.json")):
            remote = f"manifests/{paths.run_id}/{path.name}"
            _upload(path, remote)

    protected_prefixes = [
        f"interim/candidates/{paths.run_id}/",
        f"processed/{paths.run_id}/",
        f"manifests/{paths.run_id}/",
    ]
    size_mib = sum(path.stat().st_size for path in files.values()) / (1024 * 1024)
    log(f"Upload: publishing {len(files):,} files ({size_mib:.1f} MiB) to {repo_id}")
    commit_urls.append(upload_files_atomic(
        files, protected_prefixes, repo_id=repo_id, token=token,
        commit_message=f"Publish validated dataset run {paths.run_id}",
    ))
    progress.update(len(files))
    progress.finish(f"commit: {commit_urls[-1]}")
    return commit_urls


# ---------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------

def run_pipeline(
    sources: Iterable[SourceSpec],
    *,
    config: DataConfig | None = None,
    config_path: str | Path | None = None,
    work_dir: str | Path,
    run_id: str,
    append_candidates: bool = False,
    require_full_quota: bool = True,
    validate: bool = True,
    row_check: bool = True,
    upload: bool = False,
    repo_id: str = REPO_ID,
    token: str | None = None,
) -> PipelineResult:
    """
    Run the full local pipeline for one run_id.

    Status values:
        - ``complete``: quotas filled and validation passed
        - ``quota_shortfall``: processed built but quotas missing
        - ``validation_failed``: quotas ok but validation failed
        - ``unvalidated``: quotas filled but validation was disabled
        - ``failed``: aborted before a usable processed set
    """

    if config is None:
        config = load_data_config(config_path)

    paths = PipelinePaths(work_dir=Path(work_dir), run_id=run_id)
    sources = normalize_sources(sources)
    if not sources:
        raise ValueError("At least one source is required")
    for source in sources:
        if not source.archive_path.is_file():
            raise FileNotFoundError(source.archive_path)
    # Reject incompatible refill inputs before touching an existing run manifest.
    if append_candidates:
        _check_refill(config, paths, sources)
    paths.ensure()

    candidate: CandidateBuildResult | None = None
    dedup: DedupBuildResult | None = None
    processed: ProcessedBuildResult | None = None
    validation: ValidationReport | None = None
    status = "failed"

    try:
        candidate = build_candidate_pool(
            sources,
            config,
            paths,
            append=append_candidates,
        )

        dedup = build_dedup_index(paths)

        processed = build_processed_dataset(
            config,
            paths,
            require_full_quota=False,
        )
        paths.validation_manifest_path.unlink(missing_ok=True)
        write_run_manifest(
            config=config, paths=paths, candidate=candidate, dedup=dedup,
            processed=processed, validation=None, status="building",
        )

        if require_full_quota and not processed.complete:
            status = "quota_shortfall"
            write_run_manifest(
                config=config,
                paths=paths,
                candidate=candidate,
                dedup=dedup,
                processed=processed,
                validation=None,
                status=status,
            )
            raise QuotaShortfallError(processed.shortfalls)

        if validate:
            validation = validate_processed_shards(
                config,
                paths,
                row_check=row_check,
            )

        if not processed.complete:
            status = "quota_shortfall"
        elif validation is not None and not validation.ok:
            status = "validation_failed"
        elif validation is None:
            status = "unvalidated"
        else:
            status = "complete"

        manifest_path = write_run_manifest(
            config=config,
            paths=paths,
            candidate=candidate,
            dedup=dedup,
            processed=processed,
            validation=validation,
            status=status,
        )

        if upload and status == "complete":
            upload_run_artifacts(
                paths,
                repo_id=repo_id,
                token=token,
            )

        return PipelineResult(
            config=config,
            paths=paths,
            candidate=candidate,
            dedup=dedup,
            processed=processed,
            validation=validation,
            status=status,
            manifest_path=manifest_path,
        )

    except QuotaShortfallError:
        raise
    except Exception:
        if not (append_candidates and candidate is None):
            write_run_manifest(
                config=config,
                paths=paths,
                candidate=candidate,
                dedup=dedup,
                processed=processed,
                validation=validation,
                status="failed",
            )
        raise


def refill_and_rebuild(
    extra_sources: Iterable[SourceSpec],
    *,
    config: DataConfig,
    paths: PipelinePaths,
    require_full_quota: bool = True,
    validate: bool = True,
    row_check: bool = True,
) -> PipelineResult:
    """
    Append new sources to the candidate pool and rebuild.

    Rebuilds the dedup index over the entire candidate pool,
    then re-emits processed shards from scratch.
    """

    return run_pipeline(
        extra_sources,
        config=config,
        work_dir=paths.work_dir,
        run_id=paths.run_id,
        append_candidates=True,
        require_full_quota=require_full_quota,
        validate=validate,
        row_check=row_check,
    )
