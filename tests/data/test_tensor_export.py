"""Real Parquet/HDF5 conversion; only HF network operations are replaced."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import chess
import h5py
import numpy as np
import pytest

from IT3160.data import tensor_export as export
from IT3160.data.action_encoding import encode_move, legal_action_ids
from IT3160.data.canonical import encode_board
from IT3160.data.deduplication import POSITION_HASH_VERSION, position_hash
from IT3160.data.phase import classify_phase
from IT3160.data.pipeline import DataConfig, QuotaConfig
from IT3160.data.schema import ProcessedRecord, Split
from IT3160.data.split import SplitRatios, assign_split
from IT3160.data.storage import compute_checksum, write_shard


@pytest.fixture
def remote(tmp_path, monkeypatch):
    root = tmp_path / "remote"
    root.mkdir()
    config = DataConfig(split=SplitRatios(1 / 3, 1 / 3, 1 / 3), quota=QuotaConfig(2, 2, 2))
    rows = {}
    shards = []
    after_e4 = chess.Board()
    after_e4.push_uci("e2e4")
    middle = "r3k2r/ppp2ppp/2n1bn2/3qp3/3QP3/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 20"
    positions = [
        (chess.STARTING_FEN, "e2e4"), (after_e4.fen(), "e7e5"),
        (middle, "e1g1"), (middle.replace(" w ", " b "), "e8c8"),
        ("4k3/P7/8/8/8/8/7p/4K3 w - - 0 30", "a7a8n"),
        ("4k3/7P/8/8/8/8/p7/4K3 b - - 0 30", "a2a1q"),
    ]
    for split in Split:
        game_id = next(f"game-{i}" for i in range(1000)
                       if assign_split(f"game-{i}", ratios=config.split) == split)
        rows[split] = []
        for fen, uci in positions:
            board = chess.Board(fen)
            rows[split].append(ProcessedRecord(
                game_id=game_id, source_month="2021-01", split=split,
                phase=classify_phase(board), ply=board.ply(), fen=fen, move_uci=uci,
                position_hash=position_hash(board), action_id=encode_move(board, chess.Move.from_uci(uci)),
                legal_action_ids=tuple(legal_action_ids(board)),
            ))
        for index in range(3):
            name = f"processed/official-v1/{split.value}/part-{index:05d}.parquet"
            info = write_shard(rows[split][index * 2:index * 2 + 2], root / name, kind="processed")
            shards.append({"remote_path": name, "rows": info.rows, "sha256": info.sha256,
                           "size_bytes": info.size_bytes, "local_path": "/other/machine.parquet"})
    validation = {"ok": True, "error_count": 0, "checked": 18, "stats": {"row_check": True}}
    report = root / "manifests/official-v1/validation.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(validation))
    config_dict = config.to_dict()
    manifest = {
        "run_id": "official-v1", "status": "complete", "config": config_dict,
        "config_sha256": hashlib.sha256(json.dumps(
            config_dict, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "versions": {"schema_version": "1", "position_hash_version": POSITION_HASH_VERSION},
        "manifest_checksums": {"validation.json": compute_checksum(report)},
        "validation": validation,
        "processed": {"complete": True, "shard_infos": shards,
                      "selected_counts": {f"{s.value}/{p}": 2 for s in Split for p in export.PHASE_IDS}},
    }
    downloads = []
    commits = []
    state = SimpleNamespace(root=root, config=config, rows=rows, manifest=manifest,
                            head="a" * 40, downloads=downloads, commits=commits)

    def save():
        (root / "manifests/official-v1/run.json").write_text(json.dumps(manifest))

    state.save = save
    save()

    class API:
        def repo_info(self, **kwargs):
            return SimpleNamespace(sha=state.head)

        def list_repo_files(self, **kwargs):
            return [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]

        def create_commit(self, **kwargs):
            commits.append(kwargs)
            return SimpleNamespace(commit_url="https://example.test/commit/one")

        def get_paths_info(self, paths, **kwargs):
            return [SimpleNamespace(path=p, lfs={"sha256": compute_checksum(root / p)}) for p in paths]

    def download(**kwargs):
        downloads.append(kwargs)
        return str(root / kwargs["filename"])

    monkeypatch.setattr(export, "HfApi", API)
    monkeypatch.setattr(export, "hf_hub_download", download)
    return state


def run(remote, tmp_path, **kwargs):
    return export.export_tensors(work_dir=tmp_path / "output", batch_size=4, **kwargs)


def test_export_preserves_every_row_label_mask_and_special_move(remote, tmp_path):
    result = run(remote, tmp_path)
    assert len(remote.commits) == 1
    commit = remote.commits[0]
    assert commit["parent_commit"] == "a" * 40
    assert {op.path_in_repo for op in commit["operations"]} == {
        f"tensors/official-v1/{s.value}.h5" for s in Split
    }
    assert all(d["revision"] == "a" * 40 for d in remote.downloads)
    assert not any(d["filename"].startswith(("raw/", "interim/")) for d in remote.downloads)
    for split in Split:
        with h5py.File(Path(result["directory"]) / f"{split.value}.h5", "r") as f:
            assert set(f) == {"x", "y", "legal_mask", "phase"}
            assert f["x"].shape == (6, 17, 8, 8)
            assert f["x"].dtype == np.uint8
            assert f["y"].dtype == np.int64
            assert f["legal_mask"].dtype == np.bool_
            assert f["x"].compression == "gzip"
            meta = json.loads(f.attrs["metadata"])
            assert meta["revision"] == "a" * 40
            assert all("local_path" not in s for s in meta["source_shards"])
            for index, row in enumerate(remote.rows[split]):
                np.testing.assert_array_equal(f["x"][index], encode_board(chess.Board(row.fen)))
                assert f["y"][index] == row.action_id
                assert np.flatnonzero(f["legal_mask"][index]).tolist() == list(row.legal_action_ids)
                assert f["phase"][index] == export.PHASE_IDS[row.phase.value]


def test_no_upload_and_retry_reuse_completed_files_after_main_changes(remote, tmp_path, monkeypatch):
    first = run(remote, tmp_path, upload=False)
    assert not remote.commits
    remote.head = "b" * 40
    monkeypatch.setattr(export, "_write_split", lambda *a, **kw: pytest.fail("Re-exported completed file"))
    second = run(remote, tmp_path)
    assert first["files"] == second["files"]
    assert second["origin"]["revision"] == "a" * 40
    assert remote.commits[0]["parent_commit"] == "b" * 40


@pytest.mark.parametrize("change,message", [
    (lambda m: m.update(status="unvalidated"), "full successful"),
    (lambda m: m["processed"].update(complete=False), "full successful"),
    (lambda m: m["validation"]["stats"].update(row_check=False), "full successful"),
    (lambda m: m["versions"].update(position_hash_version="unknown"), "versions"),
    (lambda m: m.update(config_sha256="0" * 64), "config checksum"),
    (lambda m: m["processed"]["selected_counts"].update({"train/opening": 1}), "quotas"),
    (lambda m: m["processed"]["shard_infos"].append(m["processed"]["shard_infos"][0]), "duplicate"),
    (lambda m: m["processed"]["shard_infos"][0].update(remote_path="../outside"), "source shard path"),
])
def test_source_manifest_failures_never_publish(remote, tmp_path, change, message):
    change(remote.manifest)
    remote.save()
    with pytest.raises(ValueError, match=message):
        run(remote, tmp_path)
    assert not remote.commits


def test_source_inventory_and_validation_tampering_are_rejected(remote, tmp_path):
    unexpected = remote.root / "processed/official-v1/train/extra.parquet"
    unexpected.write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory"):
        run(remote, tmp_path)
    unexpected.unlink()
    (remote.root / "manifests/official-v1/validation.json").write_text("{}")
    with pytest.raises(ValueError, match="Checksum"):
        run(remote, tmp_path)
    assert not remote.commits


@pytest.mark.parametrize("problem", ["checksum", "label", "mask", "split", "hash", "row_count"])
def test_bad_source_rows_never_publish_and_clean_partial_file(remote, tmp_path, problem):
    info = remote.manifest["processed"]["shard_infos"][0]
    path = remote.root / info["remote_path"]
    rows = remote.rows[Split.TRAIN][:2]
    row = rows[0]
    if problem == "checksum":
        path.write_bytes(b"corrupt")
    else:
        if problem == "label":
            row = replace(row, action_id=next(a for a in row.legal_action_ids if a != row.action_id))
        elif problem == "mask":
            row = replace(row, legal_action_ids=(row.action_id,))
        elif problem == "split":
            row = replace(row, split=Split.TEST)
        elif problem == "hash":
            row = replace(row, position_hash="wrong")
        rows = [row, rows[1]] if problem != "row_count" else [row]
        actual = write_shard(rows, path, kind="processed", overwrite=True)
        info.update(sha256=actual.sha256, size_bytes=actual.size_bytes)
        remote.save()
    with pytest.raises(ValueError):
        run(remote, tmp_path)
    assert not remote.commits
    assert not list((tmp_path / "output").rglob("*.h5.tmp"))
    assert not list((tmp_path / "output").rglob("*.h5"))


def test_corrupt_local_export_blocks_retry(remote, tmp_path):
    result = run(remote, tmp_path, upload=False)
    with h5py.File(Path(result["directory"]) / "train.h5", "r+") as f:
        f["x"][0] = 0
    with pytest.raises(ValueError, match="Checksum"):
        run(remote, tmp_path)
    assert not remote.commits


def test_upload_failure_can_retry_without_reencoding(remote, tmp_path, monkeypatch):
    real_publish = export._publish
    def fail(*args, **kwargs):
        raise RuntimeError("Upload failed")
    monkeypatch.setattr(export, "_publish", fail)
    with pytest.raises(RuntimeError, match="Upload failed"):
        run(remote, tmp_path)
    monkeypatch.setattr(export, "_publish", real_publish)
    monkeypatch.setattr(export, "_write_split", lambda *a, **kw: pytest.fail("Re-encoded"))
    run(remote, tmp_path)
    assert len(remote.commits) == 1


def test_existing_remote_export_cannot_be_overwritten(remote, tmp_path):
    folder = remote.root / "tensors/official-v1"
    folder.mkdir(parents=True)
    (folder / "train.h5").write_bytes(b"older export")
    with pytest.raises(FileExistsError, match="--export-id"):
        run(remote, tmp_path)
    assert not remote.commits


def test_identical_already_published_retry_is_noop(remote, tmp_path):
    result = run(remote, tmp_path, upload=False)
    folder = remote.root / "tensors/official-v1"
    folder.mkdir(parents=True)
    for split in Split:
        name = f"{split.value}.h5"
        (folder / name).write_bytes((Path(result["directory"]) / name).read_bytes())
    result = run(remote, tmp_path)
    assert result["commit_url"].endswith("a" * 40)
    assert not remote.commits


def test_ambiguous_run_requires_explicit_choice(remote, tmp_path):
    path = remote.root / "manifests/other/run.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="--run-id"):
        run(remote, tmp_path)
    result = run(remote, tmp_path, run_id="official-v1", upload=False)
    assert result["origin"]["run_id"] == "official-v1"


@pytest.mark.parametrize("kwargs", [{"run_id": "../outside"}, {"export_id": "../outside"},
                                    {"batch_size": 0}])
def test_invalid_arguments_fail_before_network(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(export, "HfApi", lambda: pytest.fail("Accessed network"))
    with pytest.raises(ValueError):
        export.export_tensors(work_dir=tmp_path, **kwargs)


def test_script_help_and_cli_error(remote, tmp_path, capsys):
    script = Path(__file__).resolve().parents[2] / "scripts/data/export_tensors.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--no-upload" in result.stdout
    assert export.main(["--run-id", "missing", "--work-dir", str(tmp_path)]) == 1
    assert "available runs" in capsys.readouterr().err
