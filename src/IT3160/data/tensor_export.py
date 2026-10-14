"""Convert an immutable HF processed snapshot to one HDF5 file per split.

Only this exporter uses chess. Training reads x/y/legal_mask directly with h5py.
All row order, labels and split membership are preserved, without resampling.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Sequence

import chess
import h5py
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
import numpy as np

from .canonical import encode_board
from .deduplication import POSITION_HASH_VERSION
from .pipeline import DataConfig, PhaseConfig, QuotaConfig
from .progress import Progress, log
from .schema import ACTION_SPACE_SIZE, BOARD_SHAPE, Phase, Split
from .split import SplitRatios
from .storage import REPO_ID, SCHEMA_VERSION, compute_checksum, read_manifest, read_shard
from .storage import verify_checksum, write_manifest
from .validation import validate_processed_record


FORMAT_VERSION = "chess-tensors-hdf5-v1"
PHASE_IDS = {phase.value: index for index, phase in enumerate(Phase)}
REPO_ROOT = Path(__file__).resolve().parents[3]


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"Invalid run/export ID: {value!r}")
    return value


def _download(repo_id: str, revision: str, filename: str, directory: Path) -> Path:
    return Path(hf_hub_download(
        repo_id=repo_id, repo_type="dataset", revision=revision,
        filename=filename, local_dir=directory,
    ))


def _source_manifest(repo_id: str, revision: str, run_id: str | None, directory: Path):
    """Pin before listing or downloading; never combine two remote revisions."""
    api = HfApi()
    commit = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision).sha
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("HF did not return an immutable commit SHA")
    files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=commit))
    available = sorted(
        path.split("/")[1] for path in files
        if re.fullmatch(r"manifests/[A-Za-z0-9][A-Za-z0-9._-]*/run\.json", path)
    )
    if run_id is None:
        if len(available) != 1:
            raise ValueError(f"Choose --run-id explicitly; available runs: {available}")
        run_id = available[0]
    _identifier(run_id)
    name = f"manifests/{run_id}/run.json"
    if name not in files:
        raise ValueError(f"Published run not found: {name}; available runs: {available}")
    cache = directory / commit
    manifest_path = _download(repo_id, commit, name, cache)
    manifest = read_manifest(manifest_path)
    processed = manifest.get("processed") or {}
    validation = manifest.get("validation") or {}
    if (manifest.get("run_id") != run_id or manifest.get("status") != "complete"
            or processed.get("complete") is not True or validation.get("ok") is not True
            or validation.get("error_count") != 0
            or validation.get("stats", {}).get("row_check") is not True):
        raise ValueError("Source run must have complete quotas and full successful row validation")
    versions = manifest.get("versions", {})
    if (versions.get("schema_version") != SCHEMA_VERSION
            or versions.get("position_hash_version") != POSITION_HASH_VERSION):
        raise ValueError("Source encoding/schema versions differ from this checkout")
    config = manifest["config"]
    digest = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if digest != manifest.get("config_sha256"):
        raise ValueError("Source config checksum mismatch")
    config = DataConfig(
        **{key: value for key, value in config.items() if key not in {"split", "phase", "quota"}},
        split=SplitRatios(**config["split"]), phase=PhaseConfig(**config["phase"]),
        quota=QuotaConfig(**config["quota"]),
    )
    validation_name = f"manifests/{run_id}/validation.json"
    checksum = manifest.get("manifest_checksums", {}).get("validation.json")
    if validation_name not in files or not checksum:
        raise ValueError("Source validation report/checksum is missing")
    validation_path = _download(repo_id, commit, validation_name, cache)
    verify_checksum(validation_path, checksum)
    if read_manifest(validation_path) != validation:
        raise ValueError("Source validation reports disagree")
    shards = processed.get("shard_infos", [])
    expected = set()
    counts = Counter()
    for shard in shards:
        name = shard["remote_path"]
        match = re.fullmatch(
            rf"processed/{re.escape(run_id)}/(train|validation|test)/part-[0-9]{{5}}\.parquet",
            name,
        )
        if match is None or name in expected:
            raise ValueError("Invalid or duplicate source shard path")
        rows = shard["rows"]
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError("Invalid source shard row count")
        expected.add(name)
        counts[match[1]] += rows
    actual = {name for name in files if name.startswith(f"processed/{run_id}/")}
    if actual != expected:
        raise ValueError("Published processed inventory differs from manifest")
    selected = processed["selected_counts"]
    for split in Split:
        per_phase = {p: selected.get(f"{split.value}/{p}", 0) for p in PHASE_IDS}
        quota = config.quota.for_split(split)
        if (any(n != quota for n in per_phase.values())
                or counts[split.value] != sum(per_phase.values()) or counts[split.value] == 0):
            raise ValueError(f"Source quotas/row counts disagree for {split.value}")
    if validation.get("checked") != sum(counts.values()):
        raise ValueError("Source validation row count differs from processed inventory")
    origin = {
        "format_version": FORMAT_VERSION, "repo_id": repo_id, "revision": commit,
        "run_id": run_id, "run_manifest_sha256": compute_checksum(manifest_path),
        "config_sha256": digest, "source_versions": versions,
        "board_shape": list(BOARD_SHAPE), "action_space_size": ACTION_SPACE_SIZE,
        "action_layout": "plane * 64 + canonical_from_square; channel-first (73,8,8)",
        "board_layout": "side-to-move; black rotates 180 degrees; row 0 is rank 1",
        "phase_ids": PHASE_IDS,
        "encoder_sha256": {
            name: compute_checksum(Path(__file__).with_name(name))
            for name in ("canonical.py", "action_encoding.py")
        },
    }
    return manifest, config, origin, cache


def _write_split(destination: Path, split: Split, shards: list[dict], *, repo_id: str,
                 origin: dict, config: DataConfig, cache: Path, batch_size: int) -> dict:
    """Bounded memory, stable shard/row order, atomic publication after row checks."""
    rows = sum(shard["rows"] for shard in shards)
    if rows <= 0:
        raise ValueError(f"Empty split: {split.value}")
    metadata = {**origin, "split": split.value, "rows": rows, "source_shards": shards}
    progress = Progress(f"Export {split.value}", "positions")
    counts = Counter()
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".h5.tmp", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
            # Keep even the dense legal-mask chunk below h5py's default 1 MiB cache.
            chunk_rows = min(64, rows)
            for name, shape, dtype in (
                ("x", (rows, *BOARD_SHAPE), "uint8"),
                ("y", (rows,), "int64"),
                ("legal_mask", (rows, ACTION_SPACE_SIZE), "bool"),
                ("phase", (rows,), "uint8"),
            ):
                output.create_dataset(
                    name, shape=shape, dtype=dtype, chunks=(chunk_rows, *shape[1:]),
                    compression="gzip", compression_opts=1, shuffle=True, fletcher32=True,
                )
            x = np.empty((batch_size, *BOARD_SHAPE), dtype=np.uint8)
            y = np.empty(batch_size, dtype=np.int64)
            masks = np.zeros((batch_size, ACTION_SPACE_SIZE), dtype=np.bool_)
            phase = np.empty(batch_size, dtype=np.uint8)
            offset = used = 0

            def flush():
                nonlocal offset, used
                if not used:
                    return
                for name, array in (("x", x), ("y", y), ("legal_mask", masks), ("phase", phase)):
                    output[name][offset:offset + used] = array[:used]
                offset += used
                used = 0
                masks.fill(False)
                progress.update(offset)

            for shard in shards:
                log(f"Read {shard['remote_path']}")
                path = _download(repo_id, origin["revision"], shard["remote_path"], cache)
                verify_checksum(path, shard["sha256"])
                if path.stat().st_size != shard["size_bytes"]:
                    raise ValueError("Source shard size mismatch")
                shard_rows = 0
                for record in read_shard(path, kind="processed", batch_size=batch_size):
                    if record.split != split:
                        raise ValueError("Row split differs from its source shard")
                    issues = validate_processed_record(
                        record, seed=config.seed, ratios=config.split,
                        opening_max_ply=config.phase.opening_max_ply,
                        endgame_material_max=config.phase.endgame_material_max,
                    )
                    if issues:
                        raise ValueError(f"Invalid row {record.game_id}: {issues[0].code}")
                    if offset + used >= rows or shard_rows >= shard["rows"]:
                        raise ValueError("Source has more rows than declared")
                    x[used] = encode_board(chess.Board(record.fen))
                    y[used] = record.action_id
                    masks[used, list(record.legal_action_ids)] = True
                    phase[used] = PHASE_IDS[record.phase.value]
                    counts[record.phase.value] += 1
                    used += 1
                    shard_rows += 1
                    if used == batch_size:
                        flush()
                if shard_rows != shard["rows"]:
                    raise ValueError("Source shard row count mismatch")
            flush()
            if offset != rows or any(counts[p] != config.quota.for_split(split) for p in PHASE_IDS):
                raise ValueError("Exported rows/phases differ from source quotas")
            output.attrs["complete"] = True
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    progress.finish()
    return {"rows": rows, "phase_counts": dict(counts), "sha256": compute_checksum(destination)}


def _verify_output(path: Path, info: dict, origin: dict, split: Split, batch_size: int):
    """Read back compressed chunks; check stored layout and target-mask alignment."""
    verify_checksum(path, info["sha256"])
    with h5py.File(path, "r") as source:
        metadata = json.loads(source.attrs["metadata"])
        if (not source.attrs.get("complete") or metadata != {
            **origin, "split": split.value, "rows": info["rows"],
            "source_shards": metadata.get("source_shards"),
        }):
            raise ValueError("Tensor source metadata mismatch")
        n = info["rows"]
        for name, shape, dtype in (
            ("x", (n, *BOARD_SHAPE), np.dtype("uint8")),
            ("y", (n,), np.dtype("int64")),
            ("legal_mask", (n, ACTION_SPACE_SIZE), np.dtype("bool")),
            ("phase", (n,), np.dtype("uint8")),
        ):
            if source[name].shape != shape or source[name].dtype != dtype:
                raise ValueError(f"Invalid tensor layout: {name}")
        phases = Counter()
        for start in range(0, n, batch_size):
            x = source["x"][start:start + batch_size]
            y = source["y"][start:start + batch_size]
            mask = source["legal_mask"][start:start + batch_size]
            phase = source["phase"][start:start + batch_size]
            if (np.any(x > 1) or np.any(y < 0) or np.any(y >= ACTION_SPACE_SIZE)
                    or np.any(phase >= len(PHASE_IDS))):
                raise ValueError("Invalid tensor values")
            if not mask[np.arange(len(y)), y].all():
                raise ValueError("Tensor target is missing from legal mask")
            phases.update(phase.tolist())
        if {p: phases[i] for p, i in PHASE_IDS.items()} != info["phase_counts"]:
            raise ValueError("Tensor phase counts mismatch")


def _publish(files: dict[str, Path], prefix: str, *, repo_id: str) -> str:
    """Publish all three files in one commit; an identical retry is harmless."""
    api = HfApi()
    head = api.repo_info(repo_id=repo_id, repo_type="dataset").sha
    existing = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=head))
    occupied = {name for name in existing if name == prefix or name.startswith(prefix + "/")}
    if occupied:
        if occupied == set(files):
            metadata = api.get_paths_info(
                repo_id=repo_id, repo_type="dataset", paths=list(files), revision=head,
            )
            checksums = {
                item.path: (item.lfs or {}).get("sha256") for item in metadata
            }
            if all(checksums.get(name) == compute_checksum(path) for name, path in files.items()):
                log("Identical tensor files are already published")
                return f"https://huggingface.co/datasets/{repo_id}/commit/{head}"
        raise FileExistsError(f"Published folder exists: {prefix}; choose a new --export-id")
    result = api.create_commit(
        repo_id=repo_id, repo_type="dataset", parent_commit=head,
        commit_message=f"Add training tensors: {prefix}",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                    for name, path in sorted(files.items())],
    )
    return result.commit_url


def export_tensors(*, repo_id: str = REPO_ID, revision: str = "main", run_id: str | None = None,
                   export_id: str | None = None, work_dir: Path = REPO_ROOT / "data/tensors",
                   batch_size: int = 1024, upload: bool = True) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if run_id is not None:
        _identifier(run_id)
    if export_id is not None:
        _identifier(export_id)
    work_dir = Path(work_dir).resolve()
    manifest, config, origin, cache = _source_manifest(
        repo_id, revision, run_id, work_dir / "sources",
    )
    export_id = export_id or origin["run_id"]
    output = work_dir / export_id
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "export.json"
    if report_path.exists():
        report = read_manifest(report_path)
        previous = report.get("origin", {})
        # A tensor upload changes main, but does not change its source manifests.
        if ({k: v for k, v in previous.items() if k != "revision"}
                != {k: v for k, v in origin.items() if k != "revision"}):
            raise ValueError("Existing local export belongs to a different source; use --export-id")
        origin = previous
    else:
        report = {"origin": origin, "files": {}}
    for split in Split:
        name = f"{split.value}.h5"
        destination = output / name
        if name not in report["files"]:
            if destination.exists():
                raise FileExistsError(f"Untracked local tensor file: {destination}")
            shards = sorted(
                (shard for shard in manifest["processed"]["shard_infos"]
                 if shard["remote_path"].split("/")[-2] == split.value),
                key=lambda shard: shard["remote_path"],
            )
            # Only portable shard properties belong in output metadata.
            shards = [{k: v for k, v in shard.items() if k != "local_path"} for shard in shards]
            report["files"][name] = _write_split(
                destination, split, shards, repo_id=repo_id, origin=origin,
                config=config, cache=cache, batch_size=batch_size,
            )
            write_manifest(report, report_path, overwrite=True)
        log(f"Verify {name}")
        _verify_output(destination, report["files"][name], origin, split, batch_size)
    prefix = f"tensors/{export_id}"
    if upload:
        report["commit_url"] = _publish(
            {f"{prefix}/{split.value}.h5": output / f"{split.value}.h5" for split in Split},
            prefix, repo_id=repo_id,
        )
        write_manifest(report, report_path, overwrite=True)
    return {"directory": str(output), "destination": f"{repo_id}/{prefix}", **report}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--run-id", help="Source run; auto-detected only if exactly one is published")
    parser.add_argument("--revision", default="main", help="Source revision; resolved to a commit SHA")
    parser.add_argument("--export-id", help="New tensors folder name; default: source run ID")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "data/tensors")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--no-upload", action="store_true", help="Export and verify locally only")
    args = parser.parse_args(argv)
    try:
        result = export_tensors(
            repo_id=args.repo_id, revision=args.revision, run_id=args.run_id,
            export_id=args.export_id, work_dir=args.work_dir,
            batch_size=args.batch_size, upload=not args.no_upload,
        )
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
