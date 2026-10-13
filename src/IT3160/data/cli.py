"""Offline preparation commands and explicit download/upload entry points."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Sequence

import yaml

from .pipeline import (
    DataConfig, PhaseConfig, PipelinePaths, QuotaConfig, QuotaShortfallError,
    SourceSpec, load_data_config, normalize_sources, refill_and_rebuild, run_pipeline,
    upload_run_artifacts, validate_processed_shards, verify_run_artifacts,
)
from .split import SplitRatios
from .storage import REPO_ID, read_manifest, verify_checksum

from .hf_sources import download_sources, validate_hf_source_entry


def load_sources(path: str | Path) -> list[SourceSpec]:
    """Load a source manifest. Relative archive paths resolve beside the manifest."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        raise ValueError("Source manifest must contain a sources list")
    hf_origin = raw.get("huggingface")
    if hf_origin is not None and not isinstance(hf_origin, dict):
        raise ValueError("Hugging Face origin must be a mapping")
    sources = []
    allowed = {"archive_path", "source_month", "max_games", "sha256", "url", "size_bytes",
               "hf_repo_id", "hf_path"}
    for row in raw["sources"]:
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError("Invalid source entry or unknown source keys")
        if not {"archive_path", "source_month"} <= row.keys():
            raise ValueError("Each source needs archive_path and source_month")
        if hf_origin is not None:
            validate_hf_source_entry(row, hf_origin)
        archive = Path(row["archive_path"])
        if not archive.is_absolute():
            archive = path.parent / archive
        source = SourceSpec(
            archive.resolve(), row["source_month"], row.get("max_games"),
            hf_repo_id=row.get("hf_repo_id"), hf_path=row.get("hf_path"),
        )
        if not source.archive_path.is_file():
            raise FileNotFoundError(source.archive_path)
        if "sha256" in row:
            checksum = row["sha256"]
            if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
                raise ValueError("sha256 must contain 64 hexadecimal digits")
            verify_checksum(source.archive_path, checksum)
        sources.append(source)
    if not sources:
        raise ValueError("Source list cannot be empty")
    return normalize_sources(sources)


def saved_config(paths: PipelinePaths) -> DataConfig:
    payload = read_manifest(paths.run_manifest_path)["config"]
    return DataConfig(
        **{key: value for key, value in payload.items() if key not in {"split", "phase", "quota"}},
        split=SplitRatios(**payload["split"]), phase=PhaseConfig(**payload["phase"]),
        quota=QuotaConfig(**payload["quota"]),
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and audit the IT3160 chess dataset")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Fetch prepared ZIPs from Hugging Face")
    download.add_argument("--source-config", type=Path, default=Path("configs/hf_sources.yaml"),
                          help="HF source repo/prefix config")
    download.add_argument("--repo-id", help="Override the source dataset repo")
    download.add_argument("--prefix", help="Override the source folder; empty string means repo root")
    download.add_argument("--months", nargs="+", metavar="YYYY-MM",
                          help="Optional subset; default fetches every prepared ZIP")
    download.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    download.add_argument("--manifest", type=Path, help="Default: RAW_DIR/sources.json")
    for command in ("build", "refill", "validate", "inspect", "upload"):
        sub = commands.add_parser(command)
        sub.add_argument("--work-dir", type=Path, default=Path("data/runs"))
        sub.add_argument("--run-id", required=True)
        if command in {"build", "refill"}:
            sub.add_argument("--sources", type=Path, required=True)
            sub.add_argument("--config", type=Path,
                             help="Build default: configs/data.yaml; refill default: saved config")
            sub.add_argument("--allow-partial", action="store_true",
                             help="Write a pilot even if quotas are missing; status stays quota_shortfall")
        if command == "validate":
            sub.add_argument("--quick", action="store_true",
                             help="Aggregate checks only; trusts stored processed hashes/labels")
        if command == "inspect":
            sub.add_argument("--verify", action="store_true", help="Also verify checksums")
        if command == "upload":
            sub.add_argument("--repo-id", default=REPO_ID)
            sub.add_argument("--processed-only", action="store_true",
                             help="Upload processed shards and manifests, omitting candidates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.command == "download":
            with args.source_config.open(encoding="utf-8") as stream:
                origin = yaml.safe_load(stream)
            allowed = {"repo_id", "prefix", "expected_file_count"}
            if not isinstance(origin, dict) or set(origin) - allowed:
                raise ValueError("Invalid Hugging Face source config")
            origin = dict(origin)
            if args.repo_id and args.repo_id != origin.get("repo_id"):
                origin.update(repo_id=args.repo_id, expected_file_count=None)
            for name in ("repo_id", "prefix"):
                value = getattr(args, name)
                if value is not None:
                    origin[name] = value
            if args.prefix is not None:
                origin["expected_file_count"] = None
            output = download_sources(args.months, args.raw_dir,
                                      args.manifest or args.raw_dir / "sources.json", **origin)
            print(json.dumps({"sources_manifest": str(output)}))
            return 0
        paths = PipelinePaths(args.work_dir.resolve(), args.run_id)
        if args.command in {"build", "refill"}:
            sources = load_sources(args.sources)
            if args.command == "build":
                if paths.root.exists():
                    raise FileExistsError(f"Run already exists; choose a new run-id: {paths.root}")
                config = load_data_config(args.config or "configs/data.yaml")
                result = run_pipeline(sources, config=config, work_dir=paths.work_dir,
                                      run_id=paths.run_id, require_full_quota=not args.allow_partial)
            else:
                config = load_data_config(args.config) if args.config else saved_config(paths)
                result = refill_and_rebuild(sources, config=config, paths=paths,
                                            require_full_quota=not args.allow_partial)
            print(json.dumps({"status": result.status, "manifest": str(result.manifest_path),
                              "shortfalls": result.processed.shortfalls}, sort_keys=True))
            # Partial mode tolerates quota failures, never integrity errors.
            if result.validation is not None and not result.validation.ok:
                quota_errors = sum(
                    issue.code == "quota_shortfall" for issue in result.validation.issues
                )
                if result.validation.error_count != quota_errors:
                    return 1
            return 0 if result.status == "complete" or args.allow_partial else 1
        if args.command == "validate":
            verify_run_artifacts(paths)
            report = validate_processed_shards(saved_config(paths), paths, row_check=not args.quick)
            print(json.dumps(report.to_dict(), sort_keys=True))
            return 0 if report.ok else 1
        if args.command == "inspect":
            manifest = verify_run_artifacts(paths) if args.verify else read_manifest(paths.run_manifest_path)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        if args.command == "upload":
            urls = upload_run_artifacts(paths, repo_id=args.repo_id,
                                        include_candidates=not args.processed_only)
            print(json.dumps({"commits": urls}))
            return 0
    except QuotaShortfallError as exc:
        print(json.dumps({"status": "quota_shortfall", "shortfalls": exc.shortfalls,
                          "hint": "Add new months using refill; no samples are moved across splits"}),
              file=sys.stderr)
        return 2
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, zipfile.BadZipFile,
            yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
