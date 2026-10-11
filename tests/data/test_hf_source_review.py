"""Bind ordinary Git ZIP contents to the pinned Hugging Face snapshot."""

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from IT3160.data import hf_sources
from IT3160.data.storage import read_manifest


SNAPSHOT = "7db323fdf1e342caf5bcbe438cdc4ac424b53bee"
REMOTE = "raw/lichess_elite_2020-06.zip"


def zip_payload(event="a"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("games.pgn", f'[Event "{event}"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n')
    return stream.getvalue()


def git_blob_sha(payload):
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


@pytest.fixture
def ordinary_git_hub(tmp_path, monkeypatch):
    payload = zip_payload()
    state = SimpleNamespace(
        payload=payload, cached_payload=payload, blob_id=git_blob_sha(payload), calls=[],
    )

    class Api:
        def repo_info(self, **kwargs):
            state.calls.append(("info", kwargs))
            return SimpleNamespace(sha=SNAPSHOT, siblings=[SimpleNamespace(
                rfilename=REMOTE, size=len(state.payload), lfs=None, blob_id=state.blob_id,
            )])

        def list_repo_files(self, **kwargs):
            state.calls.append(("list", kwargs))
            return [REMOTE]

    def fetch(**kwargs):
        state.calls.append(("download", kwargs))
        cached = tmp_path / "fake-hf-cache" / "archive.zip"
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(state.cached_payload)
        return str(cached)

    monkeypatch.setattr(hf_sources, "HfApi", Api)
    monkeypatch.setattr(hf_sources, "hf_hub_download", fetch)
    return state


def fetch_sources(tmp_path):
    raw_dir = tmp_path / "raw"
    return hf_sources.download_sources(
        ["2020-06"], raw_dir, raw_dir / "sources.json", revision=SNAPSHOT,
    )


def test_ordinary_git_zip_is_verified_when_fetched_and_reused(tmp_path, ordinary_git_hub):
    manifest_path = fetch_sources(tmp_path)
    manifest = read_manifest(manifest_path)
    source = manifest["sources"][0]
    assert Path(source["archive_path"]).read_bytes() == ordinary_git_hub.payload
    assert source["sha256"] == hashlib.sha256(ordinary_git_hub.payload).hexdigest()
    assert source["hf_revision"] == SNAPSHOT
    assert source["hf_path"] == REMOTE
    before = manifest_path.read_bytes()
    ordinary_git_hub.calls.clear()
    fetch_sources(tmp_path)
    assert not any(action == "download" for action, _ in ordinary_git_hub.calls)
    assert manifest_path.read_bytes() == before


def test_valid_same_size_local_git_zip_must_match_remote_blob(tmp_path, ordinary_git_hub):
    altered = zip_payload("b")
    assert len(altered) == len(ordinary_git_hub.payload)
    assert git_blob_sha(altered) != ordinary_git_hub.blob_id
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    local = raw_dir / Path(REMOTE).name
    local.write_bytes(altered)
    with pytest.raises(ValueError, match="(?i)checksum|blob|digest"):
        fetch_sources(tmp_path)
    assert local.read_bytes() == altered
    assert not (raw_dir / "sources.json").exists()
    assert not any(action == "download" for action, _ in ordinary_git_hub.calls)


def test_cached_git_zip_must_match_remote_blob_before_local_publication(tmp_path,
                                                                       ordinary_git_hub):
    ordinary_git_hub.cached_payload = zip_payload("b")
    assert len(ordinary_git_hub.cached_payload) == len(ordinary_git_hub.payload)
    with pytest.raises(ValueError, match="(?i)checksum|blob|digest"):
        fetch_sources(tmp_path)
    assert not (tmp_path / "raw" / Path(REMOTE).name).exists()
    assert not (tmp_path / "raw" / "sources.json").exists()
    assert not list((tmp_path / "raw").rglob("*.part"))


def test_git_cache_changed_during_copy_cannot_publish_a_different_snapshot(tmp_path,
                                                                         ordinary_git_hub,
                                                                         monkeypatch):
    copy_file = hf_sources.shutil.copyfileobj
    altered = zip_payload("b")

    def mutate_then_copy(source, destination, length):
        Path(source.name).write_bytes(altered)
        copy_file(source, destination, length=length)

    monkeypatch.setattr(hf_sources.shutil, "copyfileobj", mutate_then_copy)
    with pytest.raises(ValueError, match="(?i)checksum|blob|digest"):
        fetch_sources(tmp_path)
    assert not (tmp_path / "raw" / Path(REMOTE).name).exists()
    assert not (tmp_path / "raw" / "sources.json").exists()
    assert not list((tmp_path / "raw").rglob("*.part"))


@pytest.mark.parametrize("blob_id", [None, "", "bad", "g" * 40])
def test_git_zip_without_valid_remote_digest_is_rejected_before_fetch(tmp_path,
                                                                     ordinary_git_hub,
                                                                     blob_id):
    ordinary_git_hub.blob_id = blob_id
    with pytest.raises(ValueError, match="(?i)checksum|blob|digest"):
        fetch_sources(tmp_path)
    assert not any(action == "download" for action, _ in ordinary_git_hub.calls)
    assert not (tmp_path / "raw" / "sources.json").exists()
