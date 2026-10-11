#!/usr/bin/env python3
"""Download prepared ZIPs, build and validate a dataset, then publish it to HF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import yaml

from IT3160.data import cli
from IT3160.data.pipeline import PipelinePaths, load_data_config
from IT3160.data.storage import REPO_ID

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New dataset version, e.g. official-v1")
    parser.add_argument("--source-config", type=Path,
                        default=REPO_ROOT / "configs/hf_sources.yaml")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/data.yaml")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "data/runs")
    parser.add_argument("--repo-id", default=REPO_ID, help="Destination HF dataset repo")
    parser.add_argument("--no-upload", action="store_true", help="Prepare and validate locally only")
    args = parser.parse_args(argv)

    try:
        paths = PipelinePaths(args.work_dir.resolve(), args.run_id)
        if paths.root.exists():
            raise FileExistsError(f"Run already exists; choose a new run-id: {paths.root}")
        load_data_config(args.config)
        if not args.source_config.is_file():
            raise FileNotFoundError(args.source_config)
    except (ValueError, TypeError, OSError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    run_args = ["--work-dir", str(paths.work_dir), "--run-id", paths.run_id]
    steps = [
        ("Download prepared ZIPs", ["download", "--source-config", str(args.source_config),
                                    "--raw-dir", str(args.raw_dir)]),
        ("Build dataset", ["build", *run_args, "--sources", str(args.raw_dir / "sources.json"),
                            "--config", str(args.config)]),
        ("Validate dataset", ["validate", *run_args]),
    ]
    if not args.no_upload:
        steps.append(("Publish dataset to Hugging Face",
                      ["upload", *run_args, "--repo-id", args.repo_id]))
    for number, (label, command) in enumerate(steps, start=1):
        print(f"[{number}/{len(steps)}] {label}", flush=True)
        result = cli.main(command)
        if result:
            print(f"Stopped at: {label} (exit code {result})", file=sys.stderr)
            return result
    print(f"Completed: {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
