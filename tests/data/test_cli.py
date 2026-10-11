"""Exercise the public commands, with all network operations replaced locally."""

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest
import yaml

from IT3160.data import cli, hf_sources
from IT3160.data.pipeline import PipelinePaths, verify_run_artifacts
from IT3160.data.storage import compute_checksum, read_manifest

PGN = '[Event "fixture"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n\n'
HF_SHA = "a" * 40


def zip_bytes(text=PGN, name="games.pgn"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, text)
    return stream.getvalue()


def mock_hf(monkeypatch, tmp_path, files):
    """Replace the HF client while retaining real cached and copied ZIP files."""
    calls = []

    class Api:
        def repo_info(self, **kwargs):
            calls.append(("info", kwargs))
            siblings = [SimpleNamespace(
                rfilename=name, size=len(contents),
                lfs=SimpleNamespace(sha256=hashlib.sha256(contents).hexdigest()),
            ) for name, contents in files.items()]
            return SimpleNamespace(sha=HF_SHA, siblings=siblings)

        def list_repo_files(self, **kwargs):
            calls.append(("list", kwargs))
            assert kwargs["revision"] == HF_SHA
            return list(files)

    def fetch(**kwargs):
        calls.append(("download", kwargs))
        assert kwargs["revision"] == HF_SHA
        cache = tmp_path / "fake-cache" / kwargs["filename"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(files[kwargs["filename"]])
        return str(cache)

    monkeypatch.setattr(hf_sources, "HfApi", Api)
    monkeypatch.setattr(hf_sources, "hf_hub_download", fetch)
    return calls


@pytest.fixture
def inputs(tmp_path):
    archive = tmp_path / "games.zip"
    archive.write_bytes(zip_bytes())
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump({"sources": [{
        "source_month": "2021-12", "archive_path": "games.zip",
        "sha256": compute_checksum(archive),
    }]}))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"quota": {
        "train_per_phase": 1, "validation_per_phase": 1, "test_per_phase": 1,
    }, "shard_size": 2}))
    return archive, sources, config


def build_args(tmp_path, inputs, *extra):
    return ["build", "--sources", str(inputs[1]), "--config", str(inputs[2]),
            "--work-dir", str(tmp_path / "runs"), "--run-id", "pilot", *extra]


def run_args(tmp_path, command, *extra):
    return [command, "--work-dir", str(tmp_path / "runs"), "--run-id", "pilot", *extra]


def test_source_manifest_resolves_relative_paths_and_verifies_hash(inputs):
    specs = cli.load_sources(inputs[1])
    assert specs[0].archive_path == inputs[0].resolve()
    assert specs[0].source_month == "2021-12"


@pytest.mark.parametrize("payload", [None, [], {}, {"sources": []}, {"sources": [None]},
    {"sources": [{"source_month": "2021-12"}]},
    {"sources": [{"archive_path": "x", "source_month": "2021-12", "typo": 1}]},
])
def test_bad_source_manifests_fail(tmp_path, payload):
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError):
        cli.load_sources(path)


def test_invalid_yaml_is_a_command_error(tmp_path, capsys):
    path = tmp_path / "malformed.yaml"
    path.write_text("sources: [unterminated")
    assert cli.main(["build", "--sources", str(path), "--run-id", "bad"]) == 1
    assert "Error:" in capsys.readouterr().err


@pytest.mark.parametrize("checksum", [False, 0, "bad", "0" * 64])
def test_bad_or_mismatched_source_checksum(inputs, checksum):
    inputs[1].write_text(yaml.safe_dump({"sources": [{
        "archive_path": "games.zip", "source_month": "2021-12", "sha256": checksum,
    }]}))
    with pytest.raises(ValueError):
        cli.load_sources(inputs[1])


def test_download_atomic_checksum_and_reuse(tmp_path, monkeypatch):
    calls = mock_hf(monkeypatch, tmp_path, {
        "raw/lichess_elite_2020-06.zip": zip_bytes(),
        "raw/lichess_elite_2020-07.zip": zip_bytes(),
    })
    raw = tmp_path / "raw"
    manifest = raw / "sources.json"
    cli.download_sources(["2020-06", "2020-07"], raw, manifest)
    downloads = [kwargs for action, kwargs in calls if action == "download"]
    assert [kwargs["filename"] for kwargs in downloads] == [
        "raw/lichess_elite_2020-06.zip", "raw/lichess_elite_2020-07.zip"]
    assert all(kwargs["repo_type"] == "dataset" for kwargs in downloads)
    assert len(cli.load_sources(manifest)) == 2
    cli.download_sources(["2020-06"], raw, manifest)
    assert len([action for action, _ in calls if action == "download"]) == 2
    assert len(read_manifest(manifest)["sources"]) == 2
    assert not list(raw.glob("*.part"))
    (raw / "lichess_elite_2020-06.zip").write_bytes(b"modified")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        cli.download_sources(["2020-06"], raw, manifest)


@pytest.mark.parametrize("payload", [b"not a ZIP", zip_bytes(name="readme.txt")])
def test_failed_download_publishes_no_partial_file(tmp_path, monkeypatch, payload):
    mock_hf(monkeypatch, tmp_path, {"raw/lichess_elite_2020-06.zip": payload})
    raw = tmp_path / "raw"
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        cli.download_sources(["2020-06"], raw, raw / "sources.json")
    assert not list(raw.rglob("*.zip"))
    assert not (raw / "sources.json").exists()
    assert not list(raw.rglob("*.part"))


def test_download_network_error_is_clear(tmp_path, monkeypatch, capsys):
    class OfflineApi:
        def repo_info(self, **kwargs):
            raise OSError("offline fixture")
    monkeypatch.setattr(hf_sources, "HfApi", OfflineApi)
    assert cli.main(["download", "--months", "2020-06", "--raw-dir", str(tmp_path / "raw")]) == 1
    assert "offline fixture" in capsys.readouterr().err


@pytest.mark.parametrize("months", [[], ["2020-13"], ["bad"], ["2020-06", "2020-06"]])
def test_invalid_months_rejected_before_download(tmp_path, monkeypatch, months):
    monkeypatch.setattr(hf_sources, "HfApi", lambda: pytest.fail("network used"))
    with pytest.raises(ValueError):
        cli.download_sources(months, tmp_path / "raw", tmp_path / "sources.json")


def test_download_cli_defaults_to_all_ten_hf_archives(tmp_path, monkeypatch, capsys):
    calls = []

    def fetch(months, raw_dir, manifest_path, **kwargs):
        calls.append((months, raw_dir, manifest_path, kwargs))
        return manifest_path

    monkeypatch.setattr(cli, "download_sources", fetch)
    assert cli.main(["download", "--raw-dir", str(tmp_path / "raw")]) == 0
    months, raw_dir, manifest, kwargs = calls[0]
    assert months is None
    assert manifest == raw_dir / "sources.json"
    assert kwargs["repo_id"] == "vohuutridung/IT3160_data"
    assert kwargs["prefix"] == "raw"
    assert kwargs["expected_file_count"] == 10
    assert json.loads(capsys.readouterr().out)["sources_manifest"] == str(manifest)


def test_download_cli_hf_overrides_and_month_subset(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "download_sources", lambda *args, **kwargs:
                        calls.append((args, kwargs)) or args[2])
    assert cli.main(["download", "--months", "2020-06", "--repo-id", "owner/other",
                     "--revision", "snapshot", "--prefix", "archives",
                     "--raw-dir", str(tmp_path / "raw")]) == 0
    args, kwargs = calls[0]
    assert args[0] == ["2020-06"]
    assert kwargs["repo_id"] == "owner/other"
    assert kwargs["revision"] == "snapshot"
    assert kwargs["prefix"] == "archives"
    assert kwargs["expected_file_count"] is None


def test_strict_build_shortage_is_nonzero_and_reviewable(tmp_path, inputs, capsys):
    assert cli.main(build_args(tmp_path, inputs)) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "quota_shortfall"
    paths = PipelinePaths(tmp_path / "runs", "pilot")
    manifest = verify_run_artifacts(paths)
    assert manifest["status"] == "quota_shortfall"
    assert manifest["processed"]["shortfalls"]


def test_partial_build_inspect_validate_and_existing_run_guard(tmp_path, inputs, capsys):
    assert cli.main(build_args(tmp_path, inputs, "--allow-partial")) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "quota_shortfall"
    assert cli.main(run_args(tmp_path, "inspect", "--verify")) == 0
    assert json.loads(capsys.readouterr().out)["runtime"]["packages"]["chess"]
    assert cli.main(run_args(tmp_path, "validate")) == 1
    assert json.loads(capsys.readouterr().out)["stats"]["row_check"] is True
    paths = PipelinePaths(tmp_path / "runs", "pilot")
    before = paths.run_manifest_path.read_bytes()
    assert cli.main(build_args(tmp_path, inputs, "--allow-partial")) == 1
    assert "already exists" in capsys.readouterr().err
    assert paths.run_manifest_path.read_bytes() == before


def test_refill_preflight_error_preserves_run(tmp_path, inputs, capsys):
    assert cli.main(build_args(tmp_path, inputs, "--allow-partial")) == 0
    capsys.readouterr()
    paths = PipelinePaths(tmp_path / "runs", "pilot")
    before = paths.run_manifest_path.read_bytes()
    assert cli.main(run_args(tmp_path, "refill", "--sources", str(inputs[1]), "--allow-partial")) == 1
    assert "new source months" in capsys.readouterr().err
    assert paths.run_manifest_path.read_bytes() == before


def test_partial_run_cannot_upload(tmp_path, inputs, monkeypatch, capsys):
    assert cli.main(build_args(tmp_path, inputs, "--allow-partial")) == 0
    monkeypatch.setattr("IT3160.data.storage.HfApi", lambda: pytest.fail("network used"))
    assert cli.main(run_args(tmp_path, "upload")) == 1
    assert "full row validation" in capsys.readouterr().err


def complete_empty_quota_run(tmp_path, inputs):
    inputs[2].write_text(yaml.safe_dump({"quota": {
        "train_per_phase": 0, "validation_per_phase": 0, "test_per_phase": 0,
    }}))
    assert cli.main(build_args(tmp_path, inputs)) == 0
    return PipelinePaths(tmp_path / "runs", "pilot")


@pytest.mark.parametrize("processed_only", [False, True])
def test_upload_collects_whole_run_in_one_atomic_commit(tmp_path, inputs, monkeypatch, capsys, processed_only):
    paths = complete_empty_quota_run(tmp_path, inputs)
    capsys.readouterr()
    calls = []

    def atomic(files, protected_prefixes, **kwargs):
        calls.append((files, protected_prefixes, kwargs))
        return "https://example.test/commit/one"

    monkeypatch.setattr("IT3160.data.pipeline.upload_files_atomic", atomic)
    extra = ["--processed-only"] if processed_only else []
    assert cli.main(run_args(tmp_path, "upload", "--repo-id", "owner/dataset", *extra)) == 0
    assert json.loads(capsys.readouterr().out)["commits"] == ["https://example.test/commit/one"]
    assert len(calls) == 1
    files, prefixes, kwargs = calls[0]
    assert all(path.is_file() for path in files.values())
    assert set(files) >= {"manifests/pilot/run.json", "manifests/pilot/sources.json",
                         "manifests/pilot/validation.json"}
    assert any(path.startswith("interim/") for path in files) is not processed_only
    assert prefixes == ["interim/candidates/pilot/", "processed/pilot/", "manifests/pilot/"]
    assert kwargs["repo_id"] == "owner/dataset"
    assert read_manifest(paths.run_manifest_path)["status"] == "complete"


def test_quick_validation_prevents_upload_until_full_recheck(tmp_path, inputs, monkeypatch, capsys):
    complete_empty_quota_run(tmp_path, inputs)
    monkeypatch.setattr("IT3160.data.pipeline.upload_files_atomic", lambda *a, **kw: [])
    assert cli.main(run_args(tmp_path, "validate", "--quick")) == 0
    assert cli.main(run_args(tmp_path, "upload")) == 1
    assert "full row validation" in capsys.readouterr().err
    assert cli.main(run_args(tmp_path, "validate")) == 0
    assert cli.main(run_args(tmp_path, "upload")) == 0


def test_config_corruption_is_detected(tmp_path, inputs):
    paths = complete_empty_quota_run(tmp_path, inputs)
    manifest = read_manifest(paths.run_manifest_path)
    manifest["config"]["seed"] += 1
    paths.run_manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Config checksum"):
        verify_run_artifacts(paths)


@pytest.mark.parametrize("command", ["download", "build", "refill", "validate", "inspect", "upload"])
def test_module_subcommand_entrypoints(command, tmp_path):
    completed = subprocess.run([sys.executable, "-m", "IT3160.data", command, "--help"],
                               cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_module_entrypoint_can_run_outside_repo(tmp_path):
    completed = subprocess.run([sys.executable, "-m", "IT3160.data", "--help"],
                               cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "download" in completed.stdout
