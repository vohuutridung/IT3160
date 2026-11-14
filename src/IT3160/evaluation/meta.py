"""Per-row metadata (FEN, move, Elo, game) aligned with an exported HDF5 split.

The tensor exporter writes rows in the order of the processed shards listed in
the file's ``metadata`` attribute. Reading the same shards in the same order
recovers the source record of every row; each row is checked against the
HDF5 labels so a misalignment cannot go unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from IT3160.data.action_encoding import encode_move
from IT3160.data.storage import read_shard, verify_checksum
from IT3160.data.tensor_export import PHASE_IDS

META_COLUMNS = (
    "row_idx", "game_id", "source_month", "ply", "fen", "move_uci",
    "white_elo", "black_elo", "result",
)


def read_h5_metadata(h5_path: str | Path) -> dict:
    with h5py.File(h5_path, "r") as source:
        return json.loads(source.attrs["metadata"])


def _resolve_shard(shard: dict, metadata: dict, processed_dir: Path | None, cache_dir: Path) -> Path:
    remote = shard["remote_path"]
    if processed_dir is not None:
        # Accept either a mirror of the repo layout or a folder of one split.
        for candidate in (processed_dir / remote, processed_dir / Path(remote).name,
                          processed_dir / Path(remote).parent.name / Path(remote).name):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Shard {remote} not found under {processed_dir}")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=metadata["repo_id"], repo_type="dataset", revision=metadata["revision"],
        filename=remote, local_dir=cache_dir,
    ))


def build_meta(
    h5_path: str | Path,
    *,
    processed_dir: str | Path | None = None,
    cache_dir: str | Path = "data/meta_cache",
) -> pa.Table:
    """Return a table with one row per HDF5 row; raises on any mismatch."""
    metadata = read_h5_metadata(h5_path)
    processed_dir = Path(processed_dir) if processed_dir is not None else None
    columns: dict[str, list] = {name: [] for name in META_COLUMNS}
    with h5py.File(h5_path, "r") as source:
        y = source["y"][:]
        phase = source["phase"][:]
    row = 0
    for shard in metadata["source_shards"]:
        path = _resolve_shard(shard, metadata, processed_dir, Path(cache_dir))
        verify_checksum(path, shard["sha256"])
        for record in read_shard(path, kind="processed"):
            if row >= len(y):
                raise ValueError("Processed shards have more rows than the HDF5 file")
            if record.action_id != int(y[row]) or PHASE_IDS[record.phase.value] != int(phase[row]):
                raise ValueError(f"Row {row} does not match its source record ({record.game_id})")
            columns["row_idx"].append(row)
            for name in META_COLUMNS[1:]:
                columns[name].append(getattr(record, name))
            row += 1
    if row != len(y):
        raise ValueError(f"Processed shards have {row} rows, HDF5 has {len(y)}")
    return pa.table({
        "row_idx": pa.array(columns["row_idx"], pa.int64()),
        "game_id": pa.array(columns["game_id"], pa.string()),
        "source_month": pa.array(columns["source_month"], pa.string()),
        "ply": pa.array(columns["ply"], pa.int32()),
        "fen": pa.array(columns["fen"], pa.string()),
        "move_uci": pa.array(columns["move_uci"], pa.string()),
        "white_elo": pa.array(columns["white_elo"], pa.int32()),
        "black_elo": pa.array(columns["black_elo"], pa.int32()),
        "result": pa.array(columns["result"], pa.string()),
    })


def save_meta(table: pa.Table, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def load_meta(
    path: str | Path,
    rows: np.ndarray | None = None,
    *,
    expected_y: np.ndarray | None = None,
    total_rows: int | None = None,
) -> dict[str, np.ndarray]:
    """Columns as numpy arrays (object dtype for strings), optionally re-indexed by ``rows``.

    ``total_rows`` (row count of the evaluated HDF5) and ``expected_y`` (labels of
    ``rows``) guard against pairing the metadata of one split with another split.
    """
    table = pq.read_table(path)
    data = {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}
    n = len(data["row_idx"])
    if not np.array_equal(data["row_idx"], np.arange(n)):
        raise ValueError("Metadata rows are not in HDF5 order")
    if total_rows is not None and n != total_rows:
        raise ValueError(f"Metadata {path} has {n} rows but the evaluated split has {total_rows}")
    if rows is not None:
        rows = np.asarray(rows)
        if len(rows) and (rows.min() < 0 or rows.max() >= n):
            raise ValueError(f"Metadata {path} has {n} rows; requested row {int(rows.max())}")
        data = {name: values[rows] for name, values in data.items()}
    if expected_y is not None:
        check_meta_labels(data, expected_y)
    return data


def check_meta_labels(meta: dict[str, np.ndarray], y: np.ndarray) -> None:
    """Raise unless every (fen, move_uci) encodes to the matching label in ``y``."""
    if len(meta["fen"]) != len(y):
        raise ValueError(f"Metadata has {len(meta['fen'])} rows, predictions have {len(y)}")
    for i, (fen, uci) in enumerate(zip(meta["fen"], meta["move_uci"])):
        action = encode_move(chess.Board(fen), chess.Move.from_uci(uci))
        if action != int(y[i]):
            raise ValueError(
                f"Metadata row {int(meta['row_idx'][i])} ({uci}) encodes to action {action}, "
                f"but the evaluated label is {int(y[i])}: --meta belongs to a different split"
            )
