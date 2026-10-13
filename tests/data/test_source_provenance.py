"""The prepared HF ZIP identity survives manifest loading and the data pipeline."""

import json
import zipfile

import pytest

from IT3160.data.cli import load_sources
from IT3160.data.pipeline import DataConfig, QuotaConfig, SourceSpec, run_pipeline
from IT3160.data.storage import compute_checksum, read_manifest

def test_hf_origin_survives_build_with_prepared_2020_source(tmp_path):
    archive = tmp_path / "lichess_elite_2020-06.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("game.pgn", '[Result "1-0"]\n\n1. e4 e5 1-0\n\n')
    origin = {"repo_id": "vohuutridung/IT3160", "prefix": "raw"}
    entry = {
        "archive_path": str(archive), "source_month": "2020-06",
        "sha256": compute_checksum(archive), "size_bytes": archive.stat().st_size,
        "hf_repo_id": origin["repo_id"],
        "hf_path": "raw/lichess_elite_2020-06.zip",
    }
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"huggingface": origin, "sources": [entry]}))
    specs = load_sources(manifest)
    assert specs[0].hf_path == entry["hf_path"]
    config = DataConfig(quota=QuotaConfig(0, 0, 0))
    result = run_pipeline(specs, config=config, work_dir=tmp_path / "runs", run_id="prepared")
    assert result.status == "complete"
    row = read_manifest(result.paths.sources_manifest_path)["sources"][0]
    for key in ("hf_repo_id", "hf_path", "source_month", "sha256"):
        assert row[key] == entry[key]

    entry["hf_path"] = "raw/lichess_elite_2020-07.zip"
    manifest.write_text(json.dumps({"huggingface": origin, "sources": [entry]}))
    with pytest.raises(ValueError, match="does not match"):
        load_sources(manifest)


@pytest.mark.parametrize("fields", [
    {"hf_repo_id": "owner/data"},
    {"hf_repo_id": "owner/data", "hf_path": "../file.zip"},
    {"hf_repo_id": "owner/data", "hf_path": "/raw/file.zip"},
    {"hf_repo_id": "owner/data", "hf_path": "raw\\file.zip"},
])
def test_source_spec_rejects_incomplete_or_unsafe_hf_metadata(tmp_path, fields):
    with pytest.raises(ValueError):
        SourceSpec(tmp_path / "unused.zip", "2020-06", **fields)
