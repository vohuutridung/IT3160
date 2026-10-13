"""Audit discovery, integrity, and safe reuse of the user's HF ZIPs."""

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from IT3160.data import cli, hf_sources
from IT3160.data.storage import read_manifest

MONTHS = [f"2020-{month:02d}" for month in range(6, 13)] + ["2021-01", "2021-02", "2021-03"]


def archive_bytes():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("games.pgn", '[Event "HF fixture"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n')
    return stream.getvalue()


@pytest.fixture
def hub(tmp_path, monkeypatch):
    payload = archive_bytes()
    state = SimpleNamespace(
        files={f"raw/lichess_elite_{month}.zip": payload for month in MONTHS},
        calls=[], checksums={}, sizes={},
    )

    class Api:
        def repo_info(self, **kwargs):
            state.calls.append(("info", kwargs))
            siblings = [SimpleNamespace(
                rfilename=name, size=state.sizes.get(name, len(contents)),
                lfs=SimpleNamespace(sha256=state.checksums.get(
                    name, hashlib.sha256(contents).hexdigest())),
            ) for name, contents in state.files.items()]
            return SimpleNamespace(siblings=siblings)

    def fetch(**kwargs):
        state.calls.append(("download", kwargs))
        cached = tmp_path / "fake-hf-cache" / kwargs["filename"]
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(state.files[kwargs["filename"]])
        return str(cached)

    monkeypatch.setattr(hf_sources, "HfApi", Api)
    monkeypatch.setattr(hf_sources, "hf_hub_download", fetch)
    return state


def fetch_sources(tmp_path, months=None, **kwargs):
    return cli.download_sources(months, tmp_path / "raw", tmp_path / "raw" / "sources.json",
                                **kwargs)


def downloads(hub):
    return [kwargs for action, kwargs in hub.calls if action == "download"]


def test_default_fetch_discovers_all_ten_uploaded_archives(tmp_path, hub):
    manifest_path = fetch_sources(tmp_path, expected_file_count=10)
    manifest = read_manifest(manifest_path)
    assert [source["source_month"] for source in manifest["sources"]] == MONTHS
    assert manifest["huggingface"] == {
        "repo_id": "vohuutridung/IT3160", "prefix": "raw",
    }
    assert len(downloads(hub)) == 10
    assert hub.calls[0] == ("info", {
        "repo_id": "vohuutridung/IT3160", "repo_type": "dataset",
        "files_metadata": True, "token": None,
    })
    for _, kwargs in hub.calls[1:]:
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["repo_id"] == "vohuutridung/IT3160"
    for source in manifest["sources"]:
        assert source["hf_repo_id"] == "vohuutridung/IT3160"
        assert source["hf_path"] == f"raw/lichess_elite_{source['source_month']}.zip"
        archive = Path(source["archive_path"])
        assert archive.parent == tmp_path / "raw"
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == source["sha256"]
        assert archive.stat().st_size == source["size_bytes"]
        assert "url" not in source
    loaded = cli.load_sources(manifest_path)
    assert len(loaded) == 10
    assert all(source.hf_repo_id == "vohuutridung/IT3160" for source in loaded)
    assert [source.hf_path for source in loaded] == [row["hf_path"] for row in manifest["sources"]]


def test_month_subset_merges_new_months_for_refill(tmp_path, hub):
    manifest_path = fetch_sources(tmp_path, ["2020-07"])
    before = read_manifest(manifest_path)["sources"][0]
    hub.calls.clear()
    fetch_sources(tmp_path, ["2020-06"])
    manifest = read_manifest(manifest_path)
    assert [row["source_month"] for row in manifest["sources"]] == ["2020-06", "2020-07"]
    assert manifest["sources"][1] == before
    assert [kwargs["filename"] for kwargs in downloads(hub)] == ["raw/lichess_elite_2020-06.zip"]


def test_reuse_verifies_local_archives_without_fetching_again(tmp_path, hub):
    manifest_path = fetch_sources(tmp_path, ["2020-06"])
    before = manifest_path.read_bytes()
    hub.calls.clear()
    fetch_sources(tmp_path, ["2020-06"])
    assert not downloads(hub)
    assert manifest_path.read_bytes() == before


def test_auth_token_is_forwarded_to_metadata_and_download(tmp_path, hub):
    fetch_sources(tmp_path, ["2020-06"], token="fixture-token")
    assert all(kwargs["token"] == "fixture-token" for _, kwargs in hub.calls)


def test_nested_remote_paths_preserve_local_folders_and_ignore_other_prefixes(tmp_path, hub):
    payload = archive_bytes()
    hub.files = {
        "raw/months/lichess_elite_2020-06.zip": payload,
        "raw/README.md": b"Raw source description",
        "raw-extra/unmappable.zip": b"ignored",
        "processed/unmappable.zip": b"ignored",
    }
    manifest = read_manifest(fetch_sources(tmp_path))
    source = manifest["sources"][0]
    assert source["hf_path"] == "raw/months/lichess_elite_2020-06.zip"
    assert Path(source["archive_path"]) == tmp_path / "raw" / "months" / "lichess_elite_2020-06.zip"
    assert len(downloads(hub)) == 1


def test_empty_prefix_can_fetch_archives_stored_at_repo_root(tmp_path, hub):
    hub.files = {"lichess_elite_2020-06.zip": archive_bytes(), "README.md": b"metadata"}
    manifest = read_manifest(fetch_sources(tmp_path, prefix=""))
    assert manifest["huggingface"]["prefix"] == ""
    assert manifest["sources"][0]["source_month"] == "2020-06"


@pytest.mark.parametrize("filename", ["uploaded.zip", "lichess_elite_2020-13.zip",
                                      "lichess_elite_2020-6.zip", "lichess_elite_2020-06.pgn.zip"])
def test_unmappable_zip_name_fails_before_any_download(tmp_path, hub, filename):
    hub.files[f"raw/{filename}"] = archive_bytes()
    with pytest.raises(ValueError):
        fetch_sources(tmp_path)
    assert not downloads(hub)
    assert not (tmp_path / "raw" / "sources.json").exists()


def test_duplicate_month_in_different_remote_folders_is_ambiguous(tmp_path, hub):
    hub.files["raw/another/lichess_elite_2020-06.zip"] = archive_bytes()
    with pytest.raises(ValueError, match="(?i)duplicate|ambiguous"):
        fetch_sources(tmp_path)
    assert not downloads(hub)


def test_missing_requested_month_is_clear_and_downloads_nothing(tmp_path, hub):
    with pytest.raises(ValueError, match="2021-04"):
        fetch_sources(tmp_path, ["2021-04"])
    assert not downloads(hub)


def test_expected_count_checks_complete_inventory_even_for_subset(tmp_path, hub):
    hub.files.pop("raw/lichess_elite_2021-03.zip")
    with pytest.raises(ValueError, match="10|count"):
        fetch_sources(tmp_path, ["2020-06"], expected_file_count=10)
    assert not downloads(hub)


def test_no_zip_in_selected_prefix_fails_clearly(tmp_path, hub):
    hub.files = {"processed/output.parquet": b"not a ZIP", "raw/README.md": b"description"}
    with pytest.raises(ValueError, match="(?i)ZIP|archive|source"):
        fetch_sources(tmp_path)
    assert not downloads(hub)


@pytest.mark.parametrize("prefix", ["/raw", "../raw", "raw/../other", "raw\\months"])
def test_unsafe_prefix_rejected_before_network(tmp_path, monkeypatch, prefix):
    monkeypatch.setattr(hf_sources, "HfApi", lambda: pytest.fail("network used"))
    with pytest.raises(ValueError):
        fetch_sources(tmp_path, prefix=prefix)


@pytest.mark.parametrize("remote_path", ["raw/../lichess_elite_2020-06.zip",
                                         "raw/a/../../lichess_elite_2020-06.zip",
                                         "raw/a\\lichess_elite_2020-06.zip"])
def test_unsafe_remote_paths_cannot_escape_raw_directory(tmp_path, hub, remote_path):
    hub.files = {remote_path: archive_bytes()}
    with pytest.raises(ValueError):
        fetch_sources(tmp_path)
    assert not downloads(hub)
    assert not (tmp_path / "lichess_elite_2020-06.zip").exists()


def test_remote_lfs_checksum_is_verified_before_archive_publication(tmp_path, hub):
    hub.checksums["raw/lichess_elite_2020-06.zip"] = "0" * 64
    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch_sources(tmp_path, ["2020-06"])
    assert not (tmp_path / "raw" / "lichess_elite_2020-06.zip").exists()
    assert not (tmp_path / "raw" / "sources.json").exists()
    assert not list((tmp_path / "raw").rglob("*.part"))


def test_dict_lfs_metadata_is_supported(tmp_path, hub, monkeypatch):
    original_api = hf_sources.HfApi

    class DictMetadataApi(original_api):
        def repo_info(self, **kwargs):
            result = super().repo_info(**kwargs)
            for sibling in result.siblings:
                sibling.lfs = {"sha256": sibling.lfs.sha256}
            return result

    monkeypatch.setattr(hf_sources, "HfApi", DictMetadataApi)
    manifest = read_manifest(fetch_sources(tmp_path, ["2020-06"]))
    expected = hashlib.sha256(hub.files["raw/lichess_elite_2020-06.zip"]).hexdigest()
    assert manifest["sources"][0]["sha256"] == expected


def test_invalid_remote_lfs_checksum_fails_before_downloading(tmp_path, hub):
    hub.checksums["raw/lichess_elite_2020-06.zip"] = "malformed"
    with pytest.raises(ValueError, match="(?i)checksum"):
        fetch_sources(tmp_path, ["2020-06"])
    assert not downloads(hub)


def test_missing_remote_inventory_is_rejected_before_download(tmp_path, hub, monkeypatch):
    original_api = hf_sources.HfApi

    class MissingMetadataApi(original_api):
        def repo_info(self, **kwargs):
            result = super().repo_info(**kwargs)
            result.siblings = []
            return result

    monkeypatch.setattr(hf_sources, "HfApi", MissingMetadataApi)
    with pytest.raises(ValueError, match="(?i)ZIP|inventory"):
        fetch_sources(tmp_path, ["2020-06"])
    assert not downloads(hub)


@pytest.mark.parametrize("override", [
    {"repo_id": "../invalid"},
    {"expected_file_count": False}, {"expected_file_count": 0}, {"expected_file_count": -1},
    {"expected_file_count": "10"},
])
def test_invalid_origin_configuration_is_rejected_before_network(tmp_path, monkeypatch, override):
    monkeypatch.setattr(hf_sources, "HfApi", lambda: pytest.fail("network used"))
    with pytest.raises(ValueError):
        fetch_sources(tmp_path, **override)


def test_remote_size_is_verified_before_archive_publication(tmp_path, hub):
    hub.sizes["raw/lichess_elite_2020-06.zip"] = 1
    with pytest.raises(ValueError, match="(?i)size"):
        fetch_sources(tmp_path, ["2020-06"])
    assert not (tmp_path / "raw" / "lichess_elite_2020-06.zip").exists()


@pytest.mark.parametrize("override", [
    {"repo_id": "owner/another"}, {"prefix": "archives"},
])
def test_existing_manifest_cannot_silently_change_origin(tmp_path, hub, override):
    manifest_path = fetch_sources(tmp_path, ["2020-06"])
    before = manifest_path.read_bytes()
    with pytest.raises(ValueError):
        fetch_sources(tmp_path, ["2020-06"], **override)
    assert manifest_path.read_bytes() == before


def test_symlinked_nested_folder_cannot_publish_outside_raw_directory(tmp_path, hub):
    hub.files = {"raw/nested/lichess_elite_2020-06.zip": archive_bytes()}
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (raw_dir / "nested").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        fetch_sources(tmp_path)
    assert not list(outside.iterdir())
    assert not downloads(hub)
