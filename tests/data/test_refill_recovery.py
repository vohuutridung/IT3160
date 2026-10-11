"""Regression checks for safe refill/download retries and unique-position quotas."""

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import chess
import pytest

from IT3160.data import cli, hf_sources, pipeline
from IT3160.data.deduplication import position_hash
from IT3160.data.phase import classify_phase
from IT3160.data.schema import CandidateRecord, Phase, Split
from IT3160.data.storage import read_manifest, write_shard


def archive_bytes(moves: str) -> bytes:
    stream = io.BytesIO()
    text = f'[Event "Local recovery fixture"]\n[Result "1-0"]\n\n{moves} 1-0\n\n'
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("games.pgn", text)
    return stream.getvalue()


def snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_refill_parser_failure_after_a_shard_flush_preserves_run_and_can_retry(
    tmp_path, monkeypatch
):
    first_zip = tmp_path / "first.zip"
    first_zip.write_bytes(archive_bytes("1. e4 e5 2. Nf3 Nc6"))
    extra_zip = tmp_path / "extra.zip"
    extra_zip.write_bytes(archive_bytes("1. d4 d5 2. c4 e6"))
    config = pipeline.DataConfig(
        shard_size=1,
        quota=pipeline.QuotaConfig(0, 0, 0),
    )
    original_run = pipeline.run_pipeline(
        [pipeline.SourceSpec(first_zip, "2021-12")],
        config=config,
        work_dir=tmp_path / "work",
        run_id="recovery",
    )
    paths = original_run.paths
    assert original_run.status == "complete"
    candidate_before = snapshot_files(paths.candidates_dir)
    source_manifest_before = paths.sources_manifest_path.read_bytes()
    run_manifest_before = paths.run_manifest_path.read_bytes()
    all_artifacts_before = snapshot_files(paths.root)
    parser = pipeline.iter_valid_games
    shard_writer = pipeline.write_shard
    flushed_candidates = []

    def fail_after_first_game(*args, **kwargs):
        stream = parser(*args, **kwargs)
        try:
            yield next(stream)
            raise RuntimeError("simulated parser failure after first game")
        finally:
            stream.close()

    def observe_flush(records, destination, *args, **kwargs):
        info = shard_writer(records, destination, *args, **kwargs)
        if kwargs.get("kind") == "candidate":
            assert Path(destination).is_file()
            flushed_candidates.append(Path(destination))
        return info

    source = pipeline.SourceSpec(extra_zip, "2022-01")
    with monkeypatch.context() as patches:
        patches.setattr(pipeline, "iter_valid_games", fail_after_first_game)
        patches.setattr(pipeline, "write_shard", observe_flush)
        with pytest.raises(RuntimeError, match="simulated parser failure"):
            pipeline.refill_and_rebuild([source], config=config, paths=paths)

    # The exception occurs after real output was flushed, not during preflight.
    assert len(flushed_candidates) == 1
    assert snapshot_files(paths.candidates_dir) == candidate_before
    assert paths.sources_manifest_path.read_bytes() == source_manifest_before
    assert paths.run_manifest_path.read_bytes() == run_manifest_before
    assert snapshot_files(paths.root) == all_artifacts_before
    pipeline.verify_run_artifacts(paths, require_complete=True)

    retried = pipeline.refill_and_rebuild([source], config=config, paths=paths)
    assert retried.status == "complete"
    assert retried.candidate.games_sampled == 1
    assert retried.dedup.accepted_games == 2
    assert len(list(pipeline.iter_local_shards(paths.candidates_dir, kind="candidate"))) == 2
    sources = read_manifest(paths.sources_manifest_path)["sources"]
    assert [entry["source_month"] for entry in sources] == ["2021-12", "2022-01"]
    after = snapshot_files(paths.candidates_dir)
    assert all(after[name] == contents for name, contents in candidate_before.items())
    pipeline.verify_run_artifacts(paths, require_complete=True)


def test_missing_raw_archive_redownload_must_match_persisted_sha_before_publish(
    tmp_path, monkeypatch
):
    original_zip = archive_bytes("1. e4 e5")
    changed_zip = archive_bytes("1. d4 d5")
    assert hashlib.sha256(original_zip).digest() != hashlib.sha256(changed_zip).digest()
    # Both responses are valid PGN-containing ZIPs: integrity failure concerns
    # source identity, rather than ZIP corruption or response syntax.
    for payload in (original_zip, changed_zip):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            assert archive.testzip() is None
            assert archive.namelist() == ["games.pgn"]

    responses = iter([original_zip, changed_zip, original_zip])
    requests = []
    snapshot = "b" * 40
    remote_path = "raw/lichess_elite_2020-06.zip"

    class Api:
        def repo_info(self, **kwargs):
            return SimpleNamespace(sha=snapshot, siblings=[SimpleNamespace(
                rfilename=remote_path, size=len(original_zip),
                lfs=SimpleNamespace(sha256=hashlib.sha256(original_zip).hexdigest()),
            )])

        def list_repo_files(self, **kwargs):
            assert kwargs["revision"] == snapshot
            return [remote_path]

    def fetch(**kwargs):
        requests.append(kwargs)
        assert kwargs["revision"] == snapshot
        assert kwargs["filename"] == remote_path
        cached = tmp_path / "cached.zip"
        cached.write_bytes(next(responses))
        return str(cached)

    monkeypatch.setattr(hf_sources, "HfApi", Api)
    monkeypatch.setattr(hf_sources, "hf_hub_download", fetch)
    raw_dir = tmp_path / "raw"
    manifest = raw_dir / "sources.json"
    cli.download_sources(["2020-06"], raw_dir, manifest)
    raw_zip = raw_dir / "lichess_elite_2020-06.zip"
    assert raw_zip.read_bytes() == original_zip
    manifest_before = manifest.read_bytes()
    raw_zip.unlink()

    with pytest.raises(ValueError, match="Checksum mismatch"):
        cli.download_sources(["2020-06"], raw_dir, manifest)
    assert manifest.read_bytes() == manifest_before
    assert not raw_zip.exists()
    assert snapshot_files(raw_dir) == {"sources.json": manifest_before}
    assert len(requests) == 2

    # A matching response can be retried without changing the original source
    # identity or requiring the caller to discard the saved manifest.
    cli.download_sources(["2020-06"], raw_dir, manifest)
    assert raw_zip.read_bytes() == original_zip
    assert read_manifest(manifest)["sources"][0]["sha256"] == hashlib.sha256(original_zip).hexdigest()
    assert len(requests) == 3
    assert not list(raw_dir.glob("*.part"))


def unique_candidates() -> list[CandidateRecord]:
    records = []
    for phase in Phase:
        for index in range(8):
            board = chess.Board(None)
            board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(chess.A2 + index, chess.Piece(chess.PAWN, chess.WHITE))
            if phase == Phase.OPENING:
                board.set_piece_at(chess.B8, chess.Piece(chess.KNIGHT, chess.BLACK))
            elif phase == Phase.MIDDLEGAME:
                board.set_piece_at(chess.B1, chess.Piece(chess.ROOK, chess.WHITE))
                board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
                board.set_piece_at(chess.D8, chess.Piece(chess.QUEEN, chess.BLACK))
                board.fullmove_number = 14
            else:
                board.set_piece_at(chess.G8, chess.Piece(chess.KNIGHT, chess.BLACK))
                board.fullmove_number = 14
            assert board.is_valid()
            assert classify_phase(board) == phase
            records.append(CandidateRecord(
                game_id=f"unique-{phase.value}-{index}",
                source_month="2021-12",
                split=Split.TRAIN,
                phase=phase,
                ply=board.ply(),
                fen=board.fen(),
                move_uci=min(board.legal_moves, key=chess.Move.uci).uci(),
                result="1-0",
            ))
    assert len({position_hash(chess.Board(record.fen)) for record in records}) == 24
    return records


def declared_priority(seed: int, purpose: str, identity: str) -> str:
    payload = json.dumps(
        [pipeline.SELECTION_VERSION, seed, purpose, identity],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def manual_processed_run(tmp_path, name, candidates, config):
    paths = pipeline.PipelinePaths(tmp_path / "manual", name)
    paths.ensure()
    for index, offset in enumerate(range(0, len(candidates), 7)):
        write_shard(
            candidates[offset:offset + 7],
            paths.candidates_dir / "train" / f"part-{index:05d}.parquet",
            kind="candidate",
        )
    pipeline.build_dedup_index(paths)
    result = pipeline.build_processed_dataset(config, paths)
    assert result.complete
    return result, list(pipeline.iter_local_shards(paths.processed_dir, kind="processed"))


def test_duplicate_observations_do_not_bias_unique_position_quota_selection(tmp_path):
    candidates = unique_candidates()
    config = pipeline.DataConfig(quota=pipeline.QuotaConfig(2, 0, 0), shard_size=3)
    expected_fingerprints = set()
    duplicates = []
    for phase in Phase:
        ordered = sorted(
            [record for record in candidates if record.phase == phase],
            key=lambda record: (
                declared_priority(config.seed, "sample", position_hash(chess.Board(record.fen))),
                position_hash(chess.Board(record.fen)),
            ),
        )
        expected_fingerprints.update(position_hash(chess.Board(record.fen)) for record in ordered[:2])
        # Repeated observations from independent games must not grant the least
        # favored fingerprint extra chances to win a quota slot.
        duplicates.extend(
            replace(ordered[-1], game_id=f"repeat-{phase.value}-{index}")
            for index in range(40)
        )
        # Also change a selected fingerprint's representative: its selected
        # status stays fixed, while the representative must remain deterministic.
        duplicates.extend(
            replace(ordered[0], game_id=f"selected-repeat-{phase.value}-{index}")
            for index in range(5)
        )

    _, baseline = manual_processed_run(tmp_path, "baseline", candidates, config)
    observed = candidates + duplicates
    result, forward = manual_processed_run(tmp_path, "duplicates", observed, config)
    _, backward = manual_processed_run(tmp_path, "reversed", list(reversed(observed)), config)
    assert {row.position_hash for row in baseline} == expected_fingerprints
    assert {row.position_hash for row in forward} == expected_fingerprints
    assert {row.position_hash for row in backward} == expected_fingerprints
    assert forward == backward
    assert result.skipped_within_split_duplicate == len(duplicates)
    assert result.selected_counts == {f"train/{phase.value}": 2 for phase in Phase}
