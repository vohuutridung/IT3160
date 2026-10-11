"""Fetch the prepared monthly ZIPs from a pinned Hugging Face dataset snapshot."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Sequence

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import validate_repo_id

from .pipeline import SourceSpec
from .storage import REPO_ID, REPO_TYPE, compute_checksum, read_manifest, verify_checksum, write_manifest


def _safe_path(value: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("Hugging Face path must be a string")
    if allow_empty and value == "":
        return ""
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value
            or path.as_posix() == "."):
        raise ValueError(f"Unsafe Hugging Face path: {value}")
    return path.as_posix()


def _check_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        if not any(not info.is_dir() and info.filename.lower().endswith(".pgn")
                   for info in archive.infolist()):
            raise ValueError(f"ZIP contains no PGN: {path}")
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"ZIP checksum failed for member {corrupt}")


def validate_hf_source_entry(row: dict, origin: dict) -> None:
    """Reject inconsistent persisted provenance before reusing a source manifest."""
    if not isinstance(row, dict) or not isinstance(origin, dict):
        raise ValueError("Invalid Hugging Face source entry")
    prefix = _safe_path(origin.get("prefix"), allow_empty=True)
    if (row.get("hf_repo_id") != origin.get("repo_id")
            or row.get("hf_revision") != origin.get("revision")):
        raise ValueError("Source entry does not match the manifest's Hugging Face origin")
    SourceSpec(Path("unused.zip"), row.get("source_month"),
               hf_repo_id=row.get("hf_repo_id"), hf_path=row.get("hf_path"),
               hf_revision=row.get("hf_revision"))
    remote = _safe_path(row.get("hf_path"))
    if prefix and not remote.startswith(prefix + "/"):
        raise ValueError("Source path does not belong to the manifest's Hugging Face prefix")
    match = re.fullmatch(r"lichess_elite_([0-9]{4}-(?:0[1-9]|1[0-2]))\.zip",
                         PurePosixPath(remote).name, flags=re.IGNORECASE)
    if not match or match.group(1) != row.get("source_month"):
        raise ValueError("Source month does not match its Hugging Face ZIP path")
    if not isinstance(row.get("sha256"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", row["sha256"]):
        raise ValueError("Source entry must contain a valid SHA-256")


def _verify_git_blob(path: Path, expected: str) -> None:
    """Check regular Git objects using the blob header, rather than raw-file SHA-1."""
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected.lower():
        raise ValueError(f"Git blob checksum mismatch for {path}")


def download_sources(
    months: Sequence[str] | None,
    raw_dir: Path,
    manifest_path: Path,
    *,
    repo_id: str = REPO_ID,
    revision: str = "main",
    prefix: str = "raw",
    expected_file_count: int | None = None,
    token: str | None = None,
) -> Path:
    """Fetch all prepared ZIPs, or selected months, and write a reusable source manifest.

    Resolve the ref once, then pin inventory, metadata and downloads to that commit.
    Existing manifests keep their pinned snapshot when the caller requests main.
    Local source files are published only after SHA-256 and ZIP validation.
    """
    validate_repo_id(repo_id)
    prefix = _safe_path(prefix, allow_empty=True)
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("A Hugging Face revision is required")
    if expected_file_count is not None and (
        type(expected_file_count) is not int or expected_file_count <= 0
    ):
        raise ValueError("expected_file_count must be a positive int")
    if months is not None:
        if isinstance(months, str) or not months or len(set(months)) != len(months):
            raise ValueError("Specify distinct source months")
        for month in months:
            SourceSpec(Path("unused.zip"), month)

    raw_dir = Path(raw_dir).resolve()
    manifest_path = Path(manifest_path).resolve()
    entries: dict[str, dict] = {}
    previous_origin = None
    if manifest_path.exists():
        previous = read_manifest(manifest_path)
        previous_origin = previous.get("huggingface")
        if not isinstance(previous_origin, dict) or (
            previous_origin.get("repo_id") != repo_id or previous_origin.get("prefix") != prefix
        ):
            raise ValueError("Source manifest belongs to a different origin; use a new manifest")
        if revision == "main":
            revision = previous_origin["revision"]
        previous_rows = previous.get("sources")
        if not isinstance(previous_rows, list):
            raise ValueError("Source manifest must contain a sources list")
        for row in previous_rows:
            validate_hf_source_entry(row, previous_origin)
            if row["source_month"] in entries:
                raise ValueError("Duplicate source month in existing source manifest")
            entries[row["source_month"]] = row

    api = HfApi()
    commit = api.repo_info(repo_id=repo_id, repo_type=REPO_TYPE, revision=revision, token=token).sha
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
        raise ValueError("Hugging Face did not return a pinned commit SHA")
    if previous_origin and previous_origin["revision"] != commit:
        raise ValueError("Source snapshot changed; use a new raw directory and manifest")
    paths = api.list_repo_files(repo_id=repo_id, repo_type=REPO_TYPE, revision=commit, token=token)
    info = api.repo_info(repo_id=repo_id, repo_type=REPO_TYPE, revision=commit,
                         files_metadata=True, token=token)
    if info.sha != commit:
        raise ValueError("Hugging Face inventory revision mismatch")
    metadata = {item.rfilename: item for item in info.siblings}

    available: dict[str, tuple[str, str]] = {}
    for remote in sorted(paths):
        if prefix and not remote.startswith(prefix + "/"):
            continue
        if not remote.lower().endswith(".zip"):
            continue
        remote = _safe_path(remote)
        relative = remote[len(prefix) + 1:] if prefix else remote
        match = re.fullmatch(r"lichess_elite_([0-9]{4}-(?:0[1-9]|1[0-2]))\.zip",
                             PurePosixPath(remote).name, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Cannot identify source month from ZIP name: {remote}")
        month = match.group(1)
        if month in available:
            raise ValueError(f"Duplicate source month in Hugging Face ZIPs: {month}")
        available[month] = remote, relative
    if not available:
        raise ValueError(f"No monthly ZIP files in Hugging Face prefix {prefix!r}")
    if expected_file_count is not None and len(available) != expected_file_count:
        raise ValueError(f"Expected {expected_file_count} source ZIPs, found {len(available)}")
    chosen = sorted(available) if months is None else sorted(months)
    missing = set(chosen) - available.keys()
    if missing:
        raise ValueError(f"Requested months absent from Hugging Face snapshot: {sorted(missing)}")

    for month in chosen:
        remote, relative = available[month]
        local = raw_dir / relative
        # A local symlink must not redirect writes or checksum checks outside raw_dir.
        if not local.resolve().is_relative_to(raw_dir):
            raise ValueError("Local ZIP path resolves outside raw_dir")
        previous_entry = entries.get(month)
        meta = metadata.get(remote)
        if meta is None:
            raise ValueError(f"Missing Hugging Face metadata for {remote}")
        lfs = getattr(meta, "lfs", None)
        remote_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        expected_size = getattr(meta, "size", None)
        if lfs is not None and (not isinstance(remote_sha, str)
                               or not re.fullmatch(r"[0-9a-fA-F]{64}", remote_sha)):
            raise ValueError(f"Invalid Hugging Face LFS checksum for {remote}")
        blob_id = getattr(meta, "blob_id", None)
        if lfs is None and (not isinstance(blob_id, str)
                            or not re.fullmatch(r"[0-9a-fA-F]{40}", blob_id)):
            raise ValueError(f"Missing or invalid Git blob checksum for {remote}")

        def check_file(path: Path) -> None:
            if previous_entry:
                verify_checksum(path, previous_entry["sha256"])
            if remote_sha:
                verify_checksum(path, remote_sha)
            else:
                _verify_git_blob(path, blob_id)
            if expected_size is not None and path.stat().st_size != expected_size:
                raise ValueError(f"Size mismatch for {remote}")
            _check_archive(path)

        if local.exists():
            check_file(local)
        else:
            cached = Path(hf_hub_download(
                repo_id=repo_id, repo_type=REPO_TYPE, filename=remote,
                revision=commit, token=token, cache_dir=raw_dir / ".hf-cache",
            ))
            check_file(cached)
            local.parent.mkdir(parents=True, exist_ok=True)
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(dir=local.parent, suffix=".part", delete=False) as out:
                    temp_path = Path(out.name)
                    with cached.open("rb") as source:
                        shutil.copyfileobj(source, out, length=1024 * 1024)
                # Check the copied bytes before publishing the local source path.
                if remote_sha:
                    verify_checksum(temp_path, remote_sha)
                else:
                    _verify_git_blob(temp_path, blob_id)
                os.replace(temp_path, local)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        entries[month] = {
            "archive_path": str(local), "source_month": month,
            "sha256": compute_checksum(local), "size_bytes": local.stat().st_size,
            "hf_repo_id": repo_id, "hf_path": remote, "hf_revision": commit,
        }
        write_manifest({
            "huggingface": {"repo_id": repo_id, "revision": commit, "prefix": prefix},
            "sources": [entries[key] for key in sorted(entries)],
        }, manifest_path, overwrite=True)
    return manifest_path
