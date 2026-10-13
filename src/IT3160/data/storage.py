
"""
Storage utilities for the chess-policy dataset.
- Read/write CandidateRecord and ProcessedRecord Parquet shards.
- Write/read JSON manifests.
- Compute SHA-256 checksums.
- Upload/download files from a Hugging Face dataset repo.
- Build consistent remote dataset paths.

Build and verify artifacts locally. Publication to the Hugging Face dataset
repository is an explicit optional step after validation.
"""

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Literal
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from .schema import (
    CandidateRecord,
    ProcessedRecord,
    Phase,
    Split,
)


REPO_ID = "vohuutridung/IT3160"
REPO_TYPE = "dataset"

SCHEMA_VERSION = "1"
DEFAULT_BATCH_SIZE = 4096
DEFAULT_COMPRESSION = "zstd"

RecordKind = Literal["candidate", "processed"]
Stage = Literal["interim", "processed"]
Record = CandidateRecord | ProcessedRecord


# ---------------------------------------------------------
# 1. Explicit Parquet schemas
# ---------------------------------------------------------

COMMON_FIELDS = [
    pa.field("game_id", pa.string(), nullable=False),
    pa.field("source_month", pa.string(), nullable=False),
    pa.field("split", pa.string(), nullable=False),
    pa.field("phase", pa.string(), nullable=False),
    pa.field("ply", pa.int32(), nullable=False),
    pa.field("fen", pa.string(), nullable=False),
    pa.field("move_uci", pa.string(), nullable=False),
    pa.field("white_elo", pa.int32()),
    pa.field("black_elo", pa.int32()),
    pa.field("result", pa.string()),
]

PROCESSED_FIELDS = [
    pa.field("position_hash", pa.string(), nullable=False),
    pa.field("action_id", pa.int32(), nullable=False),
    pa.field(
        "legal_action_ids",
        pa.list_(pa.int32()),
        nullable=False,
    ),
]


def get_parquet_schema(kind: RecordKind) -> pa.Schema:
    """Return the explicit Arrow schema for a record kind."""

    if kind not in ("candidate", "processed"):
        raise ValueError(f"Unsupported record kind: {kind}")

    fields = COMMON_FIELDS.copy()

    if kind == "processed":
        fields.extend(PROCESSED_FIELDS)

    return pa.schema(
        fields,
        metadata={
            b"it3160.schema_version": SCHEMA_VERSION.encode(),
            b"it3160.record_kind": kind.encode(),
        },
    )


# ---------------------------------------------------------
# 2. Checksum and shard metadata
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ShardInfo:
    """Metadata returned after writing a Parquet shard."""

    kind: str
    rows: int
    size_bytes: int
    sha256: str
    phase_counts: dict[str, int]
    split_counts: dict[str, int]


def compute_checksum(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Compute a file's SHA-256 without loading it into RAM."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    hasher = hashlib.sha256()

    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def verify_checksum(
    path: str | Path,
    expected_sha256: str,
) -> None:
    """Raise ValueError if a file's checksum is incorrect."""

    actual = compute_checksum(path)

    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Checksum mismatch for {path}: "
            f"expected {expected_sha256}, got {actual}"
        )


# ---------------------------------------------------------
# 3. Record serialization and deserialization
# ---------------------------------------------------------

def _record_to_dict(
    record: Record,
    kind: RecordKind,
) -> dict:
    """Convert a schema dataclass to a Parquet-compatible row."""

    expected_type = (
        CandidateRecord
        if kind == "candidate"
        else ProcessedRecord
    )

    if not isinstance(record, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__}, "
            f"got {type(record).__name__}"
        )

    return record.to_dict()


def _dict_to_record(
    row: dict,
    kind: RecordKind,
) -> Record:
    """Restore enum and tuple types from an Arrow row."""

    values = dict(row)

    values["split"] = Split(values["split"])
    values["phase"] = Phase(values["phase"])

    if kind == "processed":
        values["legal_action_ids"] = tuple(
            values["legal_action_ids"]
        )

        return ProcessedRecord(**values)

    return CandidateRecord(**values)


# ---------------------------------------------------------
# 4. Streaming Parquet writing
# ---------------------------------------------------------

def write_shard(
    records: Iterable[Record],
    local_path: str | Path,
    *,
    kind: RecordKind,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compression: str = DEFAULT_COMPRESSION,
    overwrite: bool = False,
) -> ShardInfo:
    """
    Write records incrementally into a Parquet shard.

    Does not load the entire iterable into memory.

    Uses an intermediate temporary file, then atomically
    replaces the destination once writing succeeds.

    Args:
        records:
            Generator or iterable of schema records.

        local_path:
            Temporary/local output path ending in .parquet.

        kind:
            "candidate" or "processed".

        batch_size:
            Number of records written per Arrow batch.

        compression:
            Parquet compression codec, default ZSTD.

        overwrite:
            Whether an existing local shard may be replaced.

    Returns:
        ShardInfo containing row counts and checksum.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    schema = get_parquet_schema(kind)

    local_path = Path(local_path)

    if local_path.suffix != ".parquet":
        raise ValueError("Shard filename must end in .parquet")

    local_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if local_path.exists() and not overwrite:
        raise FileExistsError(local_path)

    total_rows = 0
    phase_counts = Counter()
    split_counts = Counter()

    with tempfile.NamedTemporaryFile(
        dir=local_path.parent,
        prefix=f".{local_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temp_path = Path(temporary.name)

    try:
        with pq.ParquetWriter(
            str(temp_path),
            schema=schema,
            compression=compression,
        ) as writer:

            batch = []

            def flush_batch():
                nonlocal total_rows

                if not batch:
                    return

                table = pa.Table.from_pylist(
                    batch,
                    schema=schema,
                )

                writer.write_table(table)

                total_rows += len(batch)
                batch.clear()

            for record in records:
                row = _record_to_dict(record, kind)

                phase_counts[row["phase"]] += 1
                split_counts[row["split"]] += 1

                batch.append(row)

                if len(batch) >= batch_size:
                    flush_batch()

            flush_batch()

        if total_rows == 0:
            raise ValueError(
                "Cannot create a shard with zero records"
            )

        # Publish the completed local file.
        os.replace(temp_path, local_path)

    finally:
        # Remove unfinished temporary files on failure.
        temp_path.unlink(missing_ok=True)

    return ShardInfo(
        kind=kind,
        rows=total_rows,
        size_bytes=local_path.stat().st_size,
        sha256=compute_checksum(local_path),
        phase_counts=dict(phase_counts),
        split_counts=dict(split_counts),
    )


# ---------------------------------------------------------
# 5. Streaming Parquet reading
# ---------------------------------------------------------

def read_shard(
    local_path: str | Path,
    *,
    kind: RecordKind,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[Record]:
    """
    Read a Parquet shard by record batches.

    Restores CandidateRecord or ProcessedRecord objects.

    Checks the stored schema version and record kind.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    expected_schema = get_parquet_schema(kind)

    parquet_file = pq.ParquetFile(str(local_path))

    try:
        actual_schema = parquet_file.schema_arrow

        metadata = actual_schema.metadata or {}

        if metadata.get(b"it3160.schema_version") != (
            SCHEMA_VERSION.encode()
        ):
            raise ValueError("Incompatible schema version")

        if metadata.get(b"it3160.record_kind") != kind.encode():
            raise ValueError("Incorrect Parquet record kind")

        if not actual_schema.remove_metadata().equals(
            expected_schema.remove_metadata()
        ):
            raise ValueError("Parquet column schema mismatch")

        for batch in parquet_file.iter_batches(
            batch_size=batch_size
        ):
            for row in batch.to_pylist():
                yield _dict_to_record(row, kind)

    finally:
        parquet_file.close()


# ---------------------------------------------------------
# 6. Hugging Face repository paths
# ---------------------------------------------------------

def make_shard_path(
    stage: Stage,
    run_id: str,
    split: Split,
    shard_index: int,
) -> str:
    """
    Build a consistent Hugging Face repository path.

    Examples:
        interim/candidates/v1/train/part-00000.parquet
        processed/v1/test/part-00003.parquet
    """

    if stage not in ("interim", "processed"):
        raise ValueError("Invalid dataset stage")

    if not isinstance(split, Split):
        raise TypeError("split must be a Split enum")

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        run_id,
    ):
        raise ValueError("Invalid run_id")

    if isinstance(shard_index, bool) or not isinstance(
        shard_index, int
    ):
        raise TypeError("shard_index must be an integer")

    if shard_index < 0:
        raise ValueError("shard_index must be non-negative")

    filename = f"part-{shard_index:05d}.parquet"

    if stage == "interim":
        return (
            f"interim/candidates/{run_id}/"
            f"{split.value}/{filename}"
        )

    return (
        f"processed/{run_id}/"
        f"{split.value}/{filename}"
    )


def _validate_remote_path(path_in_repo: str) -> str:
    """Reject unsafe or unexpected remote paths."""

    if not isinstance(path_in_repo, str):
        raise TypeError("path_in_repo must be a string")

    path = PurePosixPath(path_in_repo)

    if (
        not path_in_repo
        or path_in_repo.startswith("/")
        or "\\" in path_in_repo
        or ".." in path.parts
        or path.parts[0] not in {
            "raw",
            "interim",
            "processed",
            "manifests",
        }
        or len(path.parts) < 2
    ):
        raise ValueError(
            f"Invalid repository path: {path_in_repo}"
        )

    return path.as_posix()


# ---------------------------------------------------------
# 7. Hugging Face upload/download
# ---------------------------------------------------------

def upload_file(
    local_path: str | Path,
    path_in_repo: str,
    *,
    repo_id: str = REPO_ID,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """
    Upload a file to a Hugging Face dataset repository.

    The repository and its target folder must be writable.

    Authentication:
        - HF_TOKEN environment variable, or
        - huggingface_hub login, or
        - explicit token argument.

    Returns:
        The resulting commit URL.
    """

    local_path = Path(local_path)
    path_in_repo = _validate_remote_path(path_in_repo)

    if not local_path.is_file():
        raise FileNotFoundError(local_path)

    api = HfApi()

    result = api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        token=token,
        commit_message=(
            commit_message
            or f"Upload {path_in_repo}"
        ),
    )

    return result.commit_url


def upload_files_atomic(
    files: dict[str, Path],
    protected_prefixes: list[str],
    *,
    repo_id: str = REPO_ID,
    token: str | None = None,
    commit_message: str = "Publish dataset artifacts",
) -> str:
    """Publish a new run in one commit to an existing dataset repository.

    Every destination must fall under a protected run prefix. Any existing
    file in those prefixes rejects publication: published run IDs are
    immutable, so a rebuilt dataset needs a new run ID. The commit uses the
    observed repository head as its parent to reject concurrent publication.
    """

    if not files:
        raise ValueError("At least one file is required for publication")
    if not protected_prefixes:
        raise ValueError("Protected run prefixes are required")

    prefixes = {_validate_remote_path(prefix) for prefix in protected_prefixes}
    destinations: dict[str, Path] = {}
    for path_in_repo, local_path in files.items():
        destination = _validate_remote_path(path_in_repo)
        if destination in destinations:
            raise ValueError(f"Duplicate normalized repository path: {destination}")
        if not any(destination.startswith(prefix + "/") for prefix in prefixes):
            raise ValueError(f"Destination outside protected run prefixes: {destination}")
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        destinations[destination] = local_path

    api = HfApi()
    head_commit = api.repo_info(repo_id=repo_id, repo_type=REPO_TYPE, token=token).sha
    if not isinstance(head_commit, str) or not head_commit:
        raise ValueError("Repository did not return a head commit")
    existing = api.list_repo_files(
        repo_id=repo_id, repo_type=REPO_TYPE, token=token,
    )
    if any(
        path == prefix or path.startswith(prefix + "/")
        for path in existing for prefix in prefixes
    ):
        raise FileExistsError(
            "Published run prefixes already exist; choose a new run_id"
        )

    result = api.create_commit(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        token=token,
        commit_message=commit_message,
        parent_commit=head_commit,
        operations=[
            CommitOperationAdd(path_in_repo=destination, path_or_fileobj=str(local_path))
            for destination, local_path in sorted(destinations.items())
        ],
    )
    return result.commit_url


def download_file(
    path_in_repo: str,
    *,
    repo_id: str = REPO_ID,
    token: str | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """
    Download one file from the Hugging Face dataset repo.

    Downloads into the Hugging Face cache by default.

    If expected_sha256 is given, verifies downloaded bytes.

    Do not modify the returned cached file directly.
    """

    path_in_repo = _validate_remote_path(path_in_repo)

    local_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            filename=path_in_repo,
            token=token,
        )
    )

    if expected_sha256 is not None:
        verify_checksum(local_path, expected_sha256)

    return local_path


def list_remote_files(
    prefix: str,
    *,
    repo_id: str = REPO_ID,
    token: str | None = None,
) -> list[str]:
    """
    List repository files under a given directory prefix.

    Example:
        list_remote_files("processed/v1/train/")
    """

    if not prefix or prefix.startswith("/") or ".." in prefix:
        raise ValueError("Invalid prefix")

    api = HfApi()

    all_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        token=token,
    )

    prefix = prefix.rstrip("/") + "/"

    return sorted(
        path
        for path in all_files
        if path.startswith(prefix)
    )


# ---------------------------------------------------------
# 8. JSON manifests
# ---------------------------------------------------------

def write_manifest(
    manifest: dict,
    local_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a JSON manifest to a local file."""

    local_path = Path(local_path)

    if local_path.suffix != ".json":
        raise ValueError("Manifest must be a .json file")

    local_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if local_path.exists() and not overwrite:
        raise FileExistsError(local_path)

    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=local_path.parent,
            prefix=f".{local_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            temporary.write(payload)
            temporary.write("\n")

        os.replace(temp_path, local_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return local_path


def read_manifest(
    local_path: str | Path,
) -> dict:
    """Read a local JSON manifest."""

    with Path(local_path).open(
        "r",
        encoding="utf-8",
    ) as stream:
        return json.load(stream)
