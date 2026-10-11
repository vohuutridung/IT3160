"""Verify the official runner's publishing boundary without contacting HF."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import yaml

from IT3160.data import pipeline

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("data_runner", ROOT / "scripts/data/run_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(tmp_path, *extra):
    return ["--run-id", "official-v1", "--raw-dir", str(tmp_path / "raw"),
            "--work-dir", str(tmp_path / "runs"), *extra]


def test_official_runner_defaults_and_order(runner, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner.cli, "main", lambda args: calls.append(args) or 0)
    assert runner.main(arguments(tmp_path)) == 0
    assert [call[0] for call in calls] == ["download", "build", "validate", "upload"]
    assert str(ROOT / "configs/data.yaml") in calls[1]
    assert str(ROOT / "configs/hf_sources.yaml") in calls[0]
    assert str(tmp_path / "raw/sources.json") in calls[1]
    assert calls[-1][-2:] == ["--repo-id", "vohuutridung/IT3160_data"]
    assert all("--allow-partial" not in call and "--quick" not in call for call in calls)
    assert all(call[call.index("--run-id") + 1] == "official-v1" for call in calls[1:])
    assert all(str(tmp_path / "runs") in call for call in calls[1:])


@pytest.mark.parametrize("failed_stage,code", [("download", 1), ("build", 2),
                                             ("validate", 1), ("upload", 1)])
def test_failure_stops_later_stages(runner, tmp_path, monkeypatch, failed_stage, code):
    calls = []

    def invoke(args):
        calls.append(args[0])
        return code if args[0] == failed_stage else 0

    monkeypatch.setattr(runner.cli, "main", invoke)
    assert runner.main(arguments(tmp_path)) == code
    assert calls == ["download", "build", "validate", "upload"][:calls.index(failed_stage) + 1]


def test_pilot_never_uploads(runner, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner.cli, "main", lambda args: calls.append(args) or 0)
    config = ROOT / "configs/data.pilot.yaml"
    assert runner.main(arguments(tmp_path, "--config", str(config), "--no-upload")) == 0
    assert [call[0] for call in calls] == ["download", "build", "validate"]
    assert str(config) in calls[1]


@pytest.mark.parametrize("problem", ["existing_run", "invalid_run_id", "missing_config",
                                   "invalid_config", "missing_source_config"])
def test_preflight_errors_do_not_download(runner, tmp_path, monkeypatch, problem):
    calls = []
    monkeypatch.setattr(runner.cli, "main", lambda args: calls.append(args) or 0)
    args = arguments(tmp_path)
    if problem == "existing_run":
        (tmp_path / "runs/official-v1").mkdir(parents=True)
    elif problem == "invalid_run_id":
        args[1] = "../outside"
    elif problem == "missing_config":
        args.extend(["--config", str(tmp_path / "missing.yaml")])
    elif problem == "invalid_config":
        config = tmp_path / "invalid.yaml"
        config.write_text("quota: [invalid]\n")
        args.extend(["--config", str(config)])
    else:
        args.extend(["--source-config", str(tmp_path / "missing.yaml")])
    assert runner.main(args) == 1
    assert not calls


@pytest.mark.parametrize("complete", [False, True])
def test_real_local_processing_only_publishes_complete_run(runner, tmp_path, monkeypatch, complete):
    """Only the source download and remote commit are mocked; processing is real."""
    def local_download(months, raw_dir, manifest_path, **origin):
        raw_dir.mkdir(parents=True)
        archive = raw_dir / "lichess_elite_2020-06.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("fixture.pgn", '[Event "fixture"]\n[Result "1-0"]\n\n'
                            '1. e4 e5 2. Nf3 Nc6 1-0\n\n')
        manifest_path.write_text(json.dumps({"sources": [{"archive_path": str(archive),
                                                         "source_month": "2020-06"}]}))
        return manifest_path

    published = []

    def publish(files, prefixes, **kwargs):
        published.append((files, prefixes, kwargs))
        return "https://example.test/commit/one"

    monkeypatch.setattr(runner.cli, "download_sources", local_download)
    monkeypatch.setattr(pipeline, "upload_files_atomic", publish)
    quota = 0 if complete else 1
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"quota": {"train_per_phase": quota,
                                               "validation_per_phase": quota,
                                               "test_per_phase": quota}}))
    result = runner.main(arguments(tmp_path, "--config", str(config)))
    manifest = json.loads((tmp_path / "runs/official-v1/manifests/run.json").read_text())
    if complete:
        assert result == 0
        assert manifest["status"] == "complete"
        assert len(published) == 1
        files, prefixes, kwargs = published[0]
        assert prefixes == ["interim/candidates/official-v1/", "processed/official-v1/",
                            "manifests/official-v1/"]
        assert any(name.startswith("interim/candidates/official-v1/") for name in files)
        assert "manifests/official-v1/run.json" in files
        assert kwargs["repo_id"] == "vohuutridung/IT3160_data"
    else:
        assert result == 2
        assert manifest["status"] == "quota_shortfall"
        assert not published


@pytest.mark.parametrize("script", ["run_pipeline", "download", "refill", "upload"])
def test_script_help_outside_repo(script, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/data" / f"{script}.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
