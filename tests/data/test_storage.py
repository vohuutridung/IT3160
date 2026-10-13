"""Artifact contract tests; remote operations are isolated with local fakes."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from IT3160.data import storage
from IT3160.data.action_encoding import encode_move, legal_action_ids
from IT3160.data.deduplication import position_hash
from IT3160.data.schema import CandidateRecord, Phase, ProcessedRecord, Split


def candidate(game_id="game-1", split=Split.TRAIN):
    return CandidateRecord(
        game_id=game_id, source_month="2025-01", split=split,
        phase=Phase.OPENING, ply=0, fen=chess.STARTING_FEN,
        move_uci="e2e4", white_elo=2500, black_elo=2400, result="1-0",
    )


def processed(game_id="game-1", split=Split.TRAIN):
    row = candidate(game_id, split)
    values = row.to_dict()
    values.update(split=row.split, phase=row.phase)
    board = chess.Board(row.fen)
    return ProcessedRecord(
        **values, position_hash=position_hash(board),
        action_id=encode_move(board, chess.Move.from_uci(row.move_uci)),
        legal_action_ids=tuple(legal_action_ids(board)),
    )


@pytest.mark.parametrize("kind,factory", [("candidate", candidate), ("processed", processed)])
def test_streamed_shard_roundtrip_and_metadata(tmp_path, kind, factory):
    records = [factory("a"), factory("b", Split.TEST), replace(factory("c"), white_elo=None, result=None)]
    path = tmp_path / "nested" / "part.parquet"
    info = storage.write_shard((row for row in records), path, kind=kind, batch_size=2)

    assert list(storage.read_shard(path, kind=kind, batch_size=1)) == records
    assert info.rows == 3
    assert info.kind == kind
    assert info.phase_counts == {"opening": 3}
    assert info.split_counts == {"train": 2, "test": 1}
    assert info.size_bytes == path.stat().st_size
    assert info.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not list(path.parent.glob("*.tmp"))
    stored_schema = pq.read_schema(path)
    expected_schema = storage.get_parquet_schema(kind)
    assert stored_schema.equals(expected_schema)
    assert stored_schema.metadata == expected_schema.metadata


def test_schema_has_explicit_metadata_and_required_fields():
    schema = storage.get_parquet_schema("processed")
    assert schema.metadata == {b"it3160.schema_version": b"1", b"it3160.record_kind": b"processed"}
    assert schema.field("game_id").nullable is False
    assert schema.field("white_elo").nullable is True
    assert schema.field("legal_action_ids").type == pa.list_(pa.int32())
    with pytest.raises(ValueError, match="Unsupported record kind"):
        storage.get_parquet_schema("unknown")


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda s: s.remove_metadata(), "schema version"),
        (lambda s: s.with_metadata({**s.metadata, b"it3160.schema_version": b"9"}), "schema version"),
        (lambda s: s.with_metadata({**s.metadata, b"it3160.record_kind": b"processed"}), "record kind"),
        (lambda s: pa.schema([pa.field("game_id", pa.int64(), nullable=False), *list(s)[1:]], metadata=s.metadata), "column schema"),
        (lambda s: pa.schema([pa.field("game_id", pa.string()), *list(s)[1:]], metadata=s.metadata), "column schema"),
    ],
)
def test_reader_rejects_incompatible_parquet(tmp_path, mutation, message):
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.Table.from_pylist([], schema=mutation(storage.get_parquet_schema("candidate"))), path)
    with pytest.raises(ValueError, match=message):
        list(storage.read_shard(path, kind="candidate"))


def test_reader_rejects_corrupt_bytes(tmp_path):
    path = tmp_path / "corrupt.parquet"
    path.write_bytes(b"not parquet")
    with pytest.raises(pa.ArrowInvalid):
        list(storage.read_shard(path, kind="candidate"))


def test_writer_failure_preserves_existing_file_and_cleans_partial_shard(tmp_path):
    path = tmp_path / "part.parquet"
    original = b"already published"
    path.write_bytes(original)

    def broken_source():
        yield candidate()
        raise RuntimeError("upstream parse failed")

    with pytest.raises(RuntimeError, match="upstream parse"):
        storage.write_shard(broken_source(), path, kind="candidate", batch_size=1, overwrite=True)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_writer_rejects_empty_wrong_kind_and_accidental_overwrite(tmp_path):
    path = tmp_path / "part.parquet"
    with pytest.raises(ValueError, match="zero records"):
        storage.write_shard([], path, kind="candidate")
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(TypeError, match="Expected CandidateRecord"):
        storage.write_shard([processed()], path, kind="candidate")
    assert not path.exists()
    storage.write_shard([candidate()], path, kind="candidate")
    with pytest.raises(FileExistsError):
        storage.write_shard([candidate()], path, kind="candidate")


@pytest.mark.parametrize("size", [0, -1])
def test_batch_sizes_and_checksum_chunks_must_be_positive(tmp_path, size):
    with pytest.raises(ValueError, match="batch_size"):
        storage.write_shard([candidate()], tmp_path / "a.parquet", kind="candidate", batch_size=size)
    with pytest.raises(ValueError, match="batch_size"):
        list(storage.read_shard(tmp_path / "a.parquet", kind="candidate", batch_size=size))
    with pytest.raises(ValueError, match="chunk_size"):
        storage.compute_checksum(tmp_path / "a.parquet", chunk_size=size)


def test_checksum_streaming_and_verification(tmp_path):
    path = tmp_path / "bytes"
    path.write_bytes(b"a\x00z" * 250)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert storage.compute_checksum(path, chunk_size=7) == expected
    storage.verify_checksum(path, expected.upper())
    with pytest.raises(ValueError, match="Checksum mismatch"):
        storage.verify_checksum(path, "0" * 64)


def test_manifests_are_unicode_atomic_and_explicit_about_overwrite(tmp_path):
    path = tmp_path / "metadata" / "run.json"
    manifest = {"z": 2, "description": "Dữ liệu cờ vua", "rows": [1, 2]}
    assert storage.write_manifest(manifest, path) == path
    assert storage.read_manifest(path) == manifest
    assert "Dữ liệu" in path.read_text(encoding="utf-8")
    assert path.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        storage.write_manifest(manifest, path)
    before = path.read_bytes()
    with pytest.raises(TypeError):
        storage.write_manifest({"bad": object()}, path, overwrite=True)
    assert path.read_bytes() == before
    storage.write_manifest({"updated": True}, path, overwrite=True)
    assert storage.read_manifest(path) == {"updated": True}
    assert len(list(path.parent.iterdir())) == 1


def test_artifact_extensions_are_checked(tmp_path):
    with pytest.raises(ValueError, match=".parquet"):
        storage.write_shard([candidate()], tmp_path / "data.txt", kind="candidate")
    with pytest.raises(ValueError, match=".json"):
        storage.write_manifest({}, tmp_path / "data.txt")


def test_manifest_write_failure_cleans_temporary_file(tmp_path, monkeypatch):
    real_temporary_file = storage.tempfile.NamedTemporaryFile

    class BrokenWriter:
        def __init__(self, **kwargs):
            self.file = real_temporary_file(**kwargs)
            self.name = self.file.name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def write(self, text):
            self.file.write(text[:1])
            raise OSError("disk full")

    monkeypatch.setattr(storage.tempfile, "NamedTemporaryFile", BrokenWriter)
    path = tmp_path / "run.json"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(OSError, match="disk full"):
        storage.write_manifest({"rows": 5}, path, overwrite=True)
    assert path.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("stage,expected", [
    ("interim", "interim/candidates/v1/train/part-00003.parquet"),
    ("processed", "processed/v1/train/part-00003.parquet"),
])
def test_remote_shard_paths(stage, expected):
    assert storage.make_shard_path(stage, "v1", Split.TRAIN, 3) == expected


@pytest.mark.parametrize("kwargs,error", [
    ({"stage": "raw"}, ValueError), ({"run_id": "../escape"}, ValueError),
    ({"run_id": ""}, ValueError), ({"split": "train"}, TypeError),
    ({"shard_index": True}, TypeError), ({"shard_index": 1.2}, TypeError),
    ({"shard_index": -1}, ValueError),
])
def test_remote_shard_path_validation(kwargs, error):
    values = dict(stage="processed", run_id="v1", split=Split.TRAIN, shard_index=0)
    values.update(kwargs)
    with pytest.raises(error):
        storage.make_shard_path(**values)


@pytest.mark.parametrize("path", ["", "/raw/a", "raw/../a", "raw\\a", "elsewhere/a", "processed"])
def test_upload_download_reject_unsafe_paths_before_network(tmp_path, monkeypatch, path):
    def forbidden(*args, **kwargs):
        raise AssertionError("network must not be reached")

    monkeypatch.setattr(storage, "HfApi", forbidden)
    monkeypatch.setattr(storage, "hf_hub_download", forbidden)
    with pytest.raises(ValueError):
        storage.upload_file(tmp_path / "a", path)
    with pytest.raises(ValueError):
        storage.download_file(path)


def test_huggingface_adapters_forward_arguments_and_validate_download(tmp_path, monkeypatch):
    path = tmp_path / "local.json"
    path.write_text("{}", encoding="utf-8")
    calls = []

    class FakeApi:
        def upload_file(self, **kwargs):
            calls.append(("upload", kwargs))
            return SimpleNamespace(commit_url="https://example.test/commit/123")

        def list_repo_files(self, **kwargs):
            calls.append(("list", kwargs))
            return ["processed/v10/a", "processed/v1/z", "processed/v1/a", "raw/a"]

    def fake_download(**kwargs):
        calls.append(("download", kwargs))
        return str(path)

    monkeypatch.setattr(storage, "HfApi", FakeApi)
    monkeypatch.setattr(storage, "hf_hub_download", fake_download)
    assert storage.upload_file(path, "manifests/v1.json", repo_id="owner/repo", token="test-token") == "https://example.test/commit/123"
    assert calls[-1] == ("upload", {
        "path_or_fileobj": str(path), "path_in_repo": "manifests/v1.json",
        "repo_id": "owner/repo", "repo_type": "dataset", "token": "test-token",
        "commit_message": "Upload manifests/v1.json",
    })
    expected = storage.compute_checksum(path)
    assert storage.download_file("manifests/v1.json", expected_sha256=expected) == path
    assert calls[-1][1] == {
        "repo_id": storage.REPO_ID,
        "repo_type": "dataset",
        "filename": "manifests/v1.json",
        "token": None,
    }
    with pytest.raises(ValueError, match="Checksum mismatch"):
        storage.download_file("manifests/v1.json", expected_sha256="0" * 64)
    assert storage.list_remote_files("processed/v1") == ["processed/v1/a", "processed/v1/z"]


def test_atomic_publication_uses_observed_head_as_commit_parent(tmp_path, monkeypatch):
    parquet = tmp_path / "part.parquet"
    manifest = tmp_path / "run.json"
    parquet.write_bytes(b"parquet fixture")
    manifest.write_text("{}", encoding="utf-8")
    calls = []

    class FakeApi:
        def repo_info(self, **kwargs):
            calls.append(("info", kwargs))
            return SimpleNamespace(sha="current-head-sha")

        def list_repo_files(self, **kwargs):
            calls.append(("list", kwargs))
            return ["README.md", "processed/v10/train/part-00000.parquet"]

        def create_commit(self, **kwargs):
            calls.append(("commit", kwargs))
            return SimpleNamespace(commit_url="https://example.test/commit/published")

    monkeypatch.setattr(storage, "HfApi", FakeApi)
    result = storage.upload_files_atomic({
        "processed/v1/train/part-00000.parquet": parquet,
        "manifests/v1/run.json": manifest,
    }, ["processed/v1/", "interim/candidates/v1", "manifests/v1"],
       repo_id="owner/repo", token="test-token", commit_message="Publish v1")
    assert result == "https://example.test/commit/published"
    assert [name for name, _ in calls] == ["info", "list", "commit"]
    assert calls[0][1] == {"repo_id": "owner/repo", "repo_type": "dataset", "token": "test-token"}
    assert calls[1][1] == calls[0][1]
    kwargs = calls[2][1]
    assert kwargs["parent_commit"] == "current-head-sha"
    assert kwargs["commit_message"] == "Publish v1"
    assert kwargs["repo_type"] == "dataset"
    assert [(operation.path_in_repo, operation.path_or_fileobj) for operation in kwargs["operations"]] == [
        ("manifests/v1/run.json", str(manifest)),
        ("processed/v1/train/part-00000.parquet", str(parquet)),
    ]


@pytest.mark.parametrize("existing", [
    "processed/v1/train/old-part.parquet",
    "interim/candidates/v1/test/old-part.parquet",
    "manifests/v1/run.json",
    "processed/v1",
])
def test_atomic_publication_refuses_any_existing_run_prefix(tmp_path, monkeypatch, existing):
    path = tmp_path / "run.json"
    path.write_text("{}", encoding="utf-8")

    class FakeApi:
        def repo_info(self, **kwargs):
            return SimpleNamespace(sha="head")

        def list_repo_files(self, **kwargs):
            return [existing]

        def create_commit(self, **kwargs):
            pytest.fail("Existing run must not be overwritten")

    monkeypatch.setattr(storage, "HfApi", FakeApi)
    with pytest.raises(FileExistsError, match="new run_id"):
        storage.upload_files_atomic({"manifests/v1/run.json": path},
                                    ["processed/v1", "interim/candidates/v1", "manifests/v1"])


@pytest.mark.parametrize("scenario,error", [
    ("empty_files", ValueError), ("empty_prefixes", ValueError),
    ("unsafe_path", ValueError), ("unsafe_prefix", ValueError),
    ("unprotected_path", ValueError), ("missing_local", FileNotFoundError),
    ("directory_local", FileNotFoundError), ("duplicate_normalized", ValueError),
])
def test_atomic_publication_local_guards_run_before_network(tmp_path, monkeypatch, scenario, error):
    path = tmp_path / "run.json"
    path.write_text("{}", encoding="utf-8")
    files = {"manifests/v1/run.json": path}
    prefixes = ["manifests/v1"]
    if scenario == "empty_files":
        files = {}
    elif scenario == "empty_prefixes":
        prefixes = []
    elif scenario == "unsafe_path":
        files = {"manifests/v1/../run.json": path}
    elif scenario == "unsafe_prefix":
        prefixes = ["../manifests/v1"]
    elif scenario == "unprotected_path":
        files = {"processed/v1/train/part.parquet": path}
    elif scenario == "missing_local":
        files = {"manifests/v1/run.json": tmp_path / "missing"}
    elif scenario == "directory_local":
        files = {"manifests/v1/run.json": tmp_path}
    elif scenario == "duplicate_normalized":
        files["manifests//v1/run.json"] = path

    def forbidden(*args, **kwargs):
        pytest.fail("Local validation must finish before network calls")

    monkeypatch.setattr(storage, "HfApi", forbidden)
    with pytest.raises(error):
        storage.upload_files_atomic(files, prefixes)


def test_atomic_publication_requires_existing_repository_and_propagates_commit_races(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    path.write_text("{}", encoding="utf-8")

    class MissingRepo:
        def repo_info(self, **kwargs):
            raise OSError("Repository does not exist")

    monkeypatch.setattr(storage, "HfApi", MissingRepo)
    with pytest.raises(OSError, match="does not exist"):
        storage.upload_files_atomic({"manifests/v1/run.json": path}, ["manifests/v1"])

    class ConcurrentPublisher:
        def repo_info(self, **kwargs):
            return SimpleNamespace(sha="before-other-publication")

        def list_repo_files(self, **kwargs):
            return []

        def create_commit(self, **kwargs):
            assert kwargs["parent_commit"] == "before-other-publication"
            raise RuntimeError("parent commit changed")

    monkeypatch.setattr(storage, "HfApi", ConcurrentPublisher)
    with pytest.raises(RuntimeError, match="parent commit changed"):
        storage.upload_files_atomic({"manifests/v1/run.json": path}, ["manifests/v1"])

