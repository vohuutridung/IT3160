"""Local ZIP-to-Parquet integration and reproducibility tests; no remote calls."""

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import shutil
import zipfile

import chess
import chess.pgn
import pytest
import yaml

from IT3160.data import pipeline
from IT3160.data import sampling
from IT3160.data.deduplication import game_content_hash, position_hash
from IT3160.data.pgn_parser import ParsedGame, is_valid_game, make_game_id
from IT3160.data.phase import classify_phase, material_score
from IT3160.data.sampling import sample_game_candidates
from IT3160.data.schema import CandidateRecord, Phase, Split
from IT3160.data.split import SplitRatios, assign_split
from IT3160.data.storage import read_manifest, write_shard


@pytest.fixture(scope="module")
def balanced_games():
    """Construct 4 legal games per split with unique samples in all 3 phases."""
    config = small_config()
    counts = Counter()
    fingerprints = set()
    result = []
    for attempt in range(200):
        rng = random.Random(attempt)
        board = chess.Board()
        for _ in range(200):
            moves = sorted(board.legal_moves, key=chess.Move.uci)
            if not moves:
                break
            captures = [move for move in moves if board.is_capture(move)]
            pool = captures if board.ply() > 30 and captures else moves
            board.push(rng.choice(pool))
            if board.ply() > 50 and material_score(board) <= 8:
                continuations = sorted(board.legal_moves, key=chess.Move.uci)
                if continuations:
                    board.push(rng.choice(continuations))
                break
        game = chess.pgn.Game.from_board(board)
        game.headers.update(Event="Synthetic local test", Site=f"fixture-{attempt}",
                            Date="2021.12.01", WhiteElo="2500", BlackElo="2450",
                            Result="1/2-1/2")
        content_id = game_content_hash(game)
        split = assign_split(content_id, seed=config.seed, ratios=config.split)
        if counts[split] >= 4:
            continue
        parsed = ParsedGame(make_game_id(game), "2021-12", game)
        candidates = sample_game_candidates(parsed, split, seed=config.seed)
        hashes = {position_hash(chess.Board(row.fen)) for row in candidates}
        if {row.phase for row in candidates} != set(Phase) or len(hashes) != 3:
            continue
        if hashes & fingerprints:
            continue
        fingerprints.update(hashes)
        counts[split] += 1
        result.append((game, split))
        if len(result) == 12:
            assert counts == {split: 4 for split in Split}
            return result
    pytest.fail("Unable to construct the reproducible balanced fixture")


def small_config(**changes):
    fields = dict(
        seed=42, split=SplitRatios(1 / 3, 1 / 3, 1 / 3),
        phase=pipeline.PhaseConfig(), quota=pipeline.QuotaConfig(1, 1, 1),
        shard_size=2,
    )
    fields.update(changes)
    return pipeline.DataConfig(**fields)


def game_text(game):
    return game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))


def make_archive(path, games, *, duplicate=False, invalid=False):
    texts = [game_text(game) for game in games]
    if duplicate:
        # Content identity must ignore author/site metadata and archive placement.
        clone = chess.pgn.Game.from_board(games[0].end().board())
        clone.headers.update(games[0].headers)
        clone.headers["Site"] = "duplicate metadata"
        texts.append(game_text(clone))
    if invalid:
        clone = chess.pgn.Game.from_board(games[0].end().board())
        clone.headers["Result"] = "*"
        texts.append(game_text(clone))
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        # Two PGN members exercise stable member traversal and streaming.
        midpoint = max(1, len(texts) // 2)
        archive.writestr("z-games.pgn", "\n\n".join(texts[midpoint:]) + "\n\n")
        archive.writestr("a-games.pgn", "\n\n".join(texts[:midpoint]) + "\n\n")
        archive.writestr("README.txt", "Synthetic fixture; no downloaded games.")
    return path


def rows(directory, kind):
    return list(pipeline.iter_local_shards(directory, kind=kind))


def files_snapshot(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in pipeline.list_local_shard_files(directory)
    }


def full_run(tmp_path, balanced_games, **options):
    archive = make_archive(tmp_path / "raw" / "games.zip", [game for game, _ in balanced_games],
                           duplicate=True, invalid=True)
    return pipeline.run_pipeline(
        [pipeline.SourceSpec(archive, "2021-12")], config=small_config(),
        work_dir=tmp_path / "work", run_id="test-v1", **options,
    )


def test_complete_zip_to_parquet_run_has_exact_quotas_and_auditable_provenance(tmp_path, balanced_games):
    result = full_run(tmp_path, balanced_games)
    assert result.status == "complete"
    assert result.processed.complete
    assert result.processed.shortfalls == {}
    assert result.validation.ok and result.validation.checked == 9
    assert result.validation.stats["row_check"] is True
    assert result.candidate.games_seen == 13
    assert result.candidate.games_sampled == 12
    assert result.candidate.candidate_counts == {
        f"{split.value}/{phase.value}": 4 for split in Split for phase in Phase
    }
    assert result.dedup.accepted_games == 12
    assert result.dedup.indexed_candidates == 36
    assert result.dedup.stats["cross_split_position_hashes"] == 0
    output = rows(result.paths.processed_dir, "processed")
    assert Counter((row.split, row.phase) for row in output) == {
        (split, phase): 1 for split in Split for phase in Phase
    }
    assert len({row.position_hash for row in output}) == 9
    assert all(row.action_id in row.legal_action_ids for row in output)
    assert all(assign_split(row.game_id, ratios=result.config.split) == row.split for row in output)
    sources = read_manifest(result.paths.sources_manifest_path)
    source = sources["sources"][0]
    archive = Path(source["archive_path"])
    assert source["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert source["size_bytes"] == archive.stat().st_size
    assert source["source_month"] == "2021-12"
    assert source["parser"]["games_seen"] == 14
    assert source["parser"]["games_accepted"] == 13
    assert source["parser"]["games_rejected"] == 1
    manifest = pipeline.verify_run_artifacts(result.paths, require_complete=True)
    assert manifest["config"] == result.config.to_dict()
    encoded_config = json.dumps(result.config.to_dict(), sort_keys=True, separators=(",", ":"))
    assert manifest["config_sha256"] == hashlib.sha256(encoded_config.encode()).hexdigest()
    assert manifest["identity_policy"]["split_key"] == "game_content_hash"
    assert manifest["versions"]["position_hash_version"].endswith("v2")
    assert manifest["versions"]["selection_version"] == "hash-priority-shuffle-v1"
    assert all(value != "unavailable" for value in manifest["runtime"]["packages"].values())


def test_optimized_pipeline_reuses_id_and_only_trusts_parser_checked_games(
    tmp_path, balanced_games, monkeypatch
):
    hashed = []
    sampled = []
    real_hash = pipeline.game_content_hash
    real_sample = pipeline._sample_game_candidates

    def content_hash(game):
        value = real_hash(game)
        hashed.append(value)
        return value

    def no_second_hash(game):
        pytest.fail("Sampling must reuse the pipeline's content hash")

    def sample(parsed, split, **kwargs):
        assert kwargs["check_legal_moves"] is False
        assert kwargs["content_id"] == real_hash(parsed.game)
        assert is_valid_game(parsed.game)
        sampled.append(kwargs["content_id"])
        return real_sample(parsed, split, **kwargs)

    monkeypatch.setattr(pipeline, "game_content_hash", content_hash)
    monkeypatch.setattr(sampling, "game_content_hash", no_second_hash)
    monkeypatch.setattr(pipeline, "_sample_game_candidates", sample)
    result = full_run(tmp_path, balanced_games)
    assert result.status == "complete"
    assert len(hashed) == result.candidate.games_seen == 13
    assert len(sampled) == result.candidate.games_sampled == 12
    assert len(set(sampled)) == 12


def test_optimized_pipeline_matches_legacy_rows_and_parquet_bytes(
    tmp_path, balanced_games, legacy_sampler, monkeypatch
):
    optimized = full_run(tmp_path / "optimized", balanced_games)

    def sample(parsed, split, **kwargs):
        return legacy_sampler(parsed, split, **{
            key: kwargs[key] for key in ("seed", "opening_max_ply", "endgame_material_max")
        })

    monkeypatch.setattr(pipeline, "_sample_game_candidates", sample)
    legacy = full_run(tmp_path / "legacy", balanced_games)
    assert optimized.status == legacy.status == "complete"
    assert optimized.candidate.candidate_counts == legacy.candidate.candidate_counts
    assert optimized.dedup.stats == legacy.dedup.stats
    assert optimized.validation.to_dict() == legacy.validation.to_dict()
    for kind, name in (("candidate", "candidates_dir"), ("processed", "processed_dir")):
        fast_dir, old_dir = getattr(optimized.paths, name), getattr(legacy.paths, name)
        assert rows(fast_dir, kind) == rows(old_dir, kind)
        assert files_snapshot(fast_dir) == files_snapshot(old_dir)


def test_repeated_builds_have_identical_rows_order_and_parquet_hashes(tmp_path, balanced_games):
    archive = make_archive(tmp_path / "raw.zip", [game for game, _ in balanced_games])
    options = dict(config=small_config(), work_dir=tmp_path / "work")
    first = pipeline.run_pipeline([pipeline.SourceSpec(archive, "2021-12")], run_id="first", **options)
    second = pipeline.run_pipeline([pipeline.SourceSpec(archive, "2021-12")], run_id="second", **options)
    assert rows(first.paths.processed_dir, "processed") == rows(second.paths.processed_dir, "processed")
    assert files_snapshot(first.paths.processed_dir) == files_snapshot(second.paths.processed_dir)
    assert files_snapshot(first.paths.candidates_dir) == files_snapshot(second.paths.candidates_dir)
    # Rebuilding the same run removes stale output shards and remains reproducible.
    stale = first.paths.processed_dir / "train" / "part-99999.parquet"
    stale.write_bytes(b"stale")
    rebuilt = pipeline.run_pipeline([pipeline.SourceSpec(archive, "2021-12")], run_id="first", **options)
    assert not stale.exists()
    assert files_snapshot(rebuilt.paths.processed_dir) == files_snapshot(second.paths.processed_dir)


@pytest.mark.parametrize("strict", [True, False])
def test_shortfalls_are_preserved_and_never_marked_complete(tmp_path, balanced_games, strict):
    games = [game for game, split in balanced_games if split == Split.TRAIN]
    archive = make_archive(tmp_path / "train-only.zip", games)
    paths = pipeline.PipelinePaths(tmp_path / "work", "partial")
    options = dict(config=small_config(), work_dir=paths.work_dir, run_id=paths.run_id,
                   require_full_quota=strict)
    if strict:
        with pytest.raises(pipeline.QuotaShortfallError) as error:
            pipeline.run_pipeline([pipeline.SourceSpec(archive, "2021-12")], **options)
        assert error.value.shortfalls == {
            f"{split.value}/{phase.value}": 1
            for split in (Split.VALIDATION, Split.TEST) for phase in Phase
        }
    else:
        result = pipeline.run_pipeline([pipeline.SourceSpec(archive, "2021-12")], **options)
        assert result.status == "quota_shortfall"
        assert not result.processed.complete and not result.validation.ok
        assert result.validation.checked == 3
    manifest = read_manifest(paths.run_manifest_path)
    assert manifest["status"] == "quota_shortfall"
    assert not manifest["processed"]["complete"]
    assert len(rows(paths.processed_dir, "processed")) == 3
    assert all(row.split == Split.TRAIN for row in rows(paths.processed_dir, "processed"))
    assert pipeline.verify_run_artifacts(paths)["status"] == "quota_shortfall"
    with pytest.raises(ValueError, match="complete quotas"):
        pipeline.verify_run_artifacts(paths, require_complete=True)


def test_refill_appends_new_month_and_rebuilds_index_over_all_candidates(tmp_path, balanced_games):
    first_games = [game for game, split in balanced_games if split == Split.TRAIN]
    extra_games = [game for game, split in balanced_games if split != Split.TRAIN]
    first_archive = make_archive(tmp_path / "first.zip", first_games)
    # Repeating one original game in the extra month must not sample it twice.
    extra_archive = make_archive(tmp_path / "extra.zip", extra_games + first_games[:1])
    first = pipeline.run_pipeline(
        [pipeline.SourceSpec(first_archive, "2021-12")], config=small_config(),
        work_dir=tmp_path / "work", run_id="refill", require_full_quota=False,
    )
    original_candidates = files_snapshot(first.paths.candidates_dir)
    rebuilt = pipeline.refill_and_rebuild(
        [pipeline.SourceSpec(extra_archive, "2022-01")], config=first.config, paths=first.paths,
    )
    assert rebuilt.status == "complete" and rebuilt.validation.ok
    assert rebuilt.dedup.accepted_games == 12
    assert rebuilt.dedup.indexed_candidates == 36
    assert rebuilt.dedup.stats["indexed_candidate_occurrences"] == 36
    assert len(rows(rebuilt.paths.candidates_dir, "candidate")) == 36
    assert rebuilt.candidate.games_seen == 9
    assert rebuilt.candidate.games_sampled == 8
    after = files_snapshot(rebuilt.paths.candidates_dir)
    assert all(after[name] == content for name, content in original_candidates.items())
    sources = read_manifest(rebuilt.paths.sources_manifest_path)
    assert [source["source_month"] for source in sources["sources"]] == ["2021-12", "2022-01"]
    assert len(sources["candidate_shards"]) == len(after)
    accepted = read_manifest(rebuilt.paths.accepted_games_path)
    assert set(accepted["accepted_games"]) == {game_content_hash(game) for game, _ in balanced_games}
    assert {row.source_month for row in rows(rebuilt.paths.processed_dir, "processed")} == {"2021-12", "2022-01"}
    pipeline.verify_run_artifacts(rebuilt.paths, require_complete=True)


@pytest.mark.parametrize("change", ["config", "same-month", "duplicate-input-month"])
def test_refill_rejects_changed_config_or_reused_month_before_mutating_candidates(tmp_path, balanced_games, change):
    first = full_run(tmp_path, balanced_games)
    candidate_before = files_snapshot(first.paths.candidates_dir)
    sources_before = first.paths.sources_manifest_path.read_bytes()
    run_before = first.paths.run_manifest_path.read_bytes()
    processed_before = files_snapshot(first.paths.processed_dir)
    archive = make_archive(tmp_path / "extra.zip", [balanced_games[0][0]])
    config = first.config
    sources = [pipeline.SourceSpec(archive, "2022-01")]
    if change == "config":
        config = replace(config, seed=43)
        message = "original data config"
    elif change == "same-month":
        sources = [pipeline.SourceSpec(archive, "2021-12")]
        message = "new source months"
    else:
        sources.append(pipeline.SourceSpec(archive, "2022-01"))
        message = "Duplicate source_month"
    with pytest.raises(ValueError, match=message):
        pipeline.refill_and_rebuild(sources, config=config, paths=first.paths)
    assert files_snapshot(first.paths.candidates_dir) == candidate_before
    assert first.paths.sources_manifest_path.read_bytes() == sources_before
    assert first.paths.run_manifest_path.read_bytes() == run_before
    assert files_snapshot(first.paths.processed_dir) == processed_before
    pipeline.verify_run_artifacts(first.paths, require_complete=True)


@pytest.mark.parametrize("mode", ["missing-file", "empty-sources"])
def test_preflight_failure_does_not_create_run_artifacts(tmp_path, mode):
    archive = tmp_path / "bad.zip"
    sources = [] if mode == "empty-sources" else [pipeline.SourceSpec(archive, "2021-12")]
    paths = pipeline.PipelinePaths(tmp_path / "work", "failure")
    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.run_pipeline(sources, config=small_config(), work_dir=paths.work_dir, run_id=paths.run_id)
    assert not paths.root.exists()


@pytest.mark.parametrize("stage", ["parser", "encoder"])
def test_failure_manifest_records_unusable_run_and_upload_is_rejected(tmp_path, balanced_games, monkeypatch, stage):
    archive = tmp_path / "bad.zip"
    if stage == "parser":
        archive.write_bytes(b"corrupt ZIP")
        error = zipfile.BadZipFile
    else:
        make_archive(archive, [game for game, _ in balanced_games])

        def fail_encoding(candidate):
            raise RuntimeError("simulated encoding failure")

        monkeypatch.setattr(pipeline, "encode_candidate", fail_encoding)
        error = RuntimeError
    paths = pipeline.PipelinePaths(tmp_path / "work", "failure")
    with pytest.raises(error):
        pipeline.run_pipeline([pipeline.SourceSpec(archive, "2021-12")], config=small_config(),
                              work_dir=paths.work_dir, run_id=paths.run_id)
    manifest = read_manifest(paths.run_manifest_path)
    assert manifest["status"] == "failed"
    assert manifest["config"] == small_config().to_dict()
    assert manifest["processed"] is None
    with pytest.raises(ValueError, match="complete quotas"):
        pipeline.verify_run_artifacts(paths, require_complete=True)


def manual_candidates():
    """Unique, legal positions bypass PGN sampling to isolate quota ordering."""
    records = []
    for split_index, split in enumerate(Split):
        for phase in Phase:
            for index in range(6):
                board = chess.Board(None)
                board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
                board.set_piece_at(8 + split_index * 6 + index, chess.Piece(chess.PAWN, chess.WHITE))
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
                assert board.is_valid() and classify_phase(board) == phase
                move = min(board.legal_moves, key=chess.Move.uci)
                records.append(CandidateRecord(
                    game_id=f"manual-{split.value}-{phase.value}-{index}", source_month="2021-12",
                    split=split, phase=phase, ply=board.ply(), fen=board.fen(), move_uci=move.uci(),
                ))
    assert len({position_hash(chess.Board(row.fen)) for row in records}) == 54
    return records


def priority(seed, purpose, identity):
    # Explicit, versioned manifest contract; independent of pipeline._priority().
    payload = json.dumps(["hash-priority-shuffle-v1", seed, purpose, identity], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def manual_build(tmp_path, records, config, name):
    paths = pipeline.PipelinePaths(tmp_path / "manual", name)
    paths.ensure()
    # Multiple shards and reversed row order exercise iteration-order independence.
    for split in Split:
        subset = [record for record in records if record.split == split]
        for shard_index, offset in enumerate(range(0, len(subset), 4)):
            write_shard(subset[offset:offset + 4], paths.candidates_dir / split.value / f"part-{shard_index:05d}.parquet", kind="candidate")
    pipeline.build_dedup_index(paths)
    result = pipeline.build_processed_dataset(config, paths)
    assert result.complete
    return paths, rows(paths.processed_dir, "processed")


def test_quota_selection_uses_entire_pool_and_independent_shuffle_regardless_of_shard_order(tmp_path):
    candidates = manual_candidates()
    config = small_config(quota=pipeline.QuotaConfig(2, 2, 2))
    first_paths, first = manual_build(tmp_path, candidates, config, "forward")
    second_paths, second = manual_build(tmp_path, list(reversed(candidates)), config, "reverse")
    assert first == second
    assert files_snapshot(first_paths.processed_dir) == files_snapshot(second_paths.processed_dir)
    for split in Split:
        expected = []
        for phase in Phase:
            pool = [row for row in candidates if row.split == split and row.phase == phase]
            ordered = sorted(pool, key=lambda row: (
                priority(config.seed, "sample", position_hash(chess.Board(row.fen))),
                position_hash(chess.Board(row.fen)),
            ))
            expected.extend(ordered[:2])
        expected.sort(key=lambda row: (
            priority(config.seed, "shuffle", position_hash(chess.Board(row.fen))),
            position_hash(chess.Board(row.fen)),
        ))
        output = [row for row in first if row.split == split]
        assert [row.game_id for row in output] == [row.game_id for row in expected]
        assert Counter(row.phase for row in output) == {phase: 2 for phase in Phase}
    _, changed = manual_build(tmp_path, candidates, replace(config, seed=43), "new-seed")
    assert {row.game_id for row in changed} != {row.game_id for row in first}


def test_same_split_duplicates_choose_seeded_representative_before_quotas(tmp_path):
    candidates = manual_candidates()
    aliases = [replace(row, game_id=f"alias-{row.game_id}") for row in candidates[::6]]
    pool = candidates + aliases
    config = small_config(quota=pipeline.QuotaConfig(6, 6, 6))
    _, forward = manual_build(tmp_path, pool, config, "duplicate-forward")
    _, reverse = manual_build(tmp_path, list(reversed(pool)), config, "duplicate-reverse")
    assert forward == reverse
    assert len(forward) == 54
    by_hash = {row.position_hash: row for row in forward}
    assert len(by_hash) == 54
    for original, alias in zip(candidates[::6], aliases):
        fingerprint = position_hash(chess.Board(original.fen))
        expected = min((original, alias), key=lambda row: priority(
            config.seed, "representative", f"{row.game_id}/{row.ply}/{row.move_uci}",
        ))
        assert by_hash[fingerprint].game_id == expected.game_id


def test_repeated_observations_do_not_bias_position_quota_selection(tmp_path):
    candidates = manual_candidates()
    config = small_config(quota=pipeline.QuotaConfig(2, 2, 2))
    repeated = candidates + [
        replace(row, game_id=f"repeat-{repeat}-{row.game_id}")
        for repeat in range(12) for row in candidates[::6]
    ]
    _, unique_output = manual_build(tmp_path, candidates, config, "unique-pool")
    _, repeated_output = manual_build(tmp_path, repeated, config, "repeated-pool")
    assert [row.position_hash for row in unique_output] == [row.position_hash for row in repeated_output]


def test_full_index_rebuild_removes_old_position_when_new_split_conflict_arrives(tmp_path):
    candidates = manual_candidates()
    config = small_config(quota=pipeline.QuotaConfig(6, 6, 6))
    paths, original = manual_build(tmp_path, candidates, config, "new-conflict")
    target = candidates[0]
    fingerprint = position_hash(chess.Board(target.fen))
    assert fingerprint in {row.position_hash for row in original}
    conflicting = replace(target, game_id="different-game-same-position", split=Split.TEST,
                          source_month="2022-01")
    write_shard([conflicting], paths.candidates_dir / "test" / "part-00005.parquet", kind="candidate")
    rebuilt = pipeline.build_dedup_index(paths)
    assert rebuilt.indexed_candidates == 55
    assert rebuilt.stats["cross_split_position_hashes"] == 1
    output = pipeline.build_processed_dataset(config, paths, require_full_quota=False)
    assert output.skipped_cross_split == 2
    assert output.shortfalls == {"train/opening": 1}
    assert not output.complete
    assert fingerprint not in {row.position_hash for row in rows(paths.processed_dir, "processed")}
    assert len(rows(paths.processed_dir, "processed")) == 53


def test_refill_detects_candidate_checksum_damage_before_touching_manifests(tmp_path, balanced_games):
    original = full_run(tmp_path, balanced_games)
    target = pipeline.list_local_shard_files(original.paths.candidates_dir)[0]
    target.write_bytes(target.read_bytes() + b"tampered")
    candidates_before = files_snapshot(original.paths.candidates_dir)
    run_before = original.paths.run_manifest_path.read_bytes()
    sources_before = original.paths.sources_manifest_path.read_bytes()
    archive = make_archive(tmp_path / "extra.zip", [balanced_games[0][0]])
    with pytest.raises(ValueError, match="Checksum mismatch"):
        pipeline.refill_and_rebuild([pipeline.SourceSpec(archive, "2022-01")],
                                    config=original.config, paths=original.paths)
    assert files_snapshot(original.paths.candidates_dir) == candidates_before
    assert original.paths.run_manifest_path.read_bytes() == run_before
    assert original.paths.sources_manifest_path.read_bytes() == sources_before


@pytest.mark.parametrize("section, values", [
    (None, {"unknown": 1}), ("split", {"validation_typo": 0.05}),
    ("phase", {"opening_typo": 24}), ("quota", {"train_typo": 1}),
])
def test_config_loader_rejects_unknown_keys(tmp_path, section, values):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(values if section is None else {section: values}))
    with pytest.raises(ValueError, match="Unknown"):
        pipeline.load_data_config(config_file)


@pytest.mark.parametrize("path", [
    ("seed",), ("shard_size",), ("max_samples_per_game_per_phase",),
    ("phase", "opening_max_ply"), ("phase", "endgame_material_max"),
    ("quota", "train_per_phase"), ("quota", "validation_per_phase"), ("quota", "test_per_phase"),
])
@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_config_loader_preserves_and_rejects_bad_integer_types(tmp_path, path, value):
    raw = {path[0]: value} if len(path) == 1 else {path[0]: {path[1]: value}}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(raw))
    with pytest.raises((TypeError, ValueError)):
        pipeline.load_data_config(config_file)


@pytest.mark.parametrize("raw", [[], 1, "text", {"phase": []}, {"quota": None}, {"split": "bad"}])
def test_config_loader_requires_mapping_sections(tmp_path, raw):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="mapping"):
        pipeline.load_data_config(config_file)


def test_config_defaults_roundtrip_and_empty_yaml(tmp_path):
    assert pipeline.load_data_config() == pipeline.DataConfig()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    assert pipeline.load_data_config(config_file) == pipeline.DataConfig()
    custom = small_config()
    config_file.write_text(yaml.safe_dump(custom.to_dict()))
    assert pipeline.load_data_config(config_file) == custom


@pytest.mark.parametrize("month", ["2022-00", "2022-13", "２０２６-01", "2026-1", "2026-01 ", None])
def test_source_spec_enforces_ascii_month_format(tmp_path, month):
    with pytest.raises(ValueError):
        pipeline.SourceSpec(tmp_path / "raw.zip", month)


@pytest.mark.parametrize("month", ["2020-06", "2021-03", "2021-12"])
def test_source_spec_accepts_prepared_months_paths_and_zero_game_limit(tmp_path, month):
    source = pipeline.SourceSpec(str(tmp_path / "raw.zip"), month, max_games=0)
    assert isinstance(source.archive_path, Path)
    assert source.max_games == 0


@pytest.mark.parametrize("limit", [True, 1.5, "1", -1])
def test_source_spec_rejects_bad_game_limits(tmp_path, limit):
    with pytest.raises(ValueError):
        pipeline.SourceSpec(tmp_path / "raw.zip", "2021-12", max_games=limit)


@pytest.mark.parametrize("kind", ["candidate", "processed"])
@pytest.mark.parametrize("damage", ["tampered", "missing", "extra"])
def test_saved_run_verification_rejects_altered_parquet_inventory(tmp_path, balanced_games, kind, damage):
    result = full_run(tmp_path, balanced_games)
    directory = result.paths.candidates_dir if kind == "candidate" else result.paths.processed_dir
    target = pipeline.list_local_shard_files(directory)[0]
    if damage == "tampered":
        target.write_bytes(target.read_bytes() + b"corruption")
        expected = "Checksum mismatch"
    elif damage == "missing":
        target.unlink()
        expected = "inventory mismatch"
    else:
        shutil.copyfile(target, target.parent / "part-99999.parquet")
        expected = "inventory mismatch"
    with pytest.raises(ValueError, match=expected):
        pipeline.verify_run_artifacts(result.paths, require_complete=True)


@pytest.mark.parametrize("manifest_name", ["sources.json", "validation.json"])
def test_saved_run_verifies_manifest_checksums(tmp_path, balanced_games, manifest_name):
    result = full_run(tmp_path, balanced_games)
    path = result.paths.manifests_dir / manifest_name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        pipeline.verify_run_artifacts(result.paths, require_complete=True)


def test_quick_validation_cannot_qualify_dataset_for_upload(tmp_path, balanced_games):
    result = full_run(tmp_path, balanced_games, row_check=False)
    assert result.status == "complete" and result.validation.ok
    assert result.validation.stats["row_check"] is False
    with pytest.raises(ValueError, match="full row validation"):
        pipeline.verify_run_artifacts(result.paths, require_complete=True)
    report = pipeline.validate_processed_shards(result.config, result.paths, row_check=True)
    assert report.ok
    pipeline.verify_run_artifacts(result.paths, require_complete=True)


def test_skipping_validation_is_explicitly_unvalidated(tmp_path, balanced_games):
    result = full_run(tmp_path, balanced_games, validate=False)
    assert result.status == "unvalidated"
    assert result.processed.complete and result.validation is None
    assert pipeline.verify_run_artifacts(result.paths)["status"] == "unvalidated"
    with pytest.raises(ValueError, match="full row validation"):
        pipeline.verify_run_artifacts(result.paths, require_complete=True)


def test_validation_config_must_match_saved_run(tmp_path, balanced_games):
    result = full_run(tmp_path, balanced_games)
    before = result.paths.validation_manifest_path.read_bytes()
    with pytest.raises(ValueError, match="config does not match"):
        pipeline.validate_processed_shards(replace(result.config, seed=43), result.paths)
    assert result.paths.validation_manifest_path.read_bytes() == before
