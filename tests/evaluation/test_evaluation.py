import json
import math
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from IT3160.evaluation.baselines import fit_action_prior, prior_predict_fn, random_legal_summary
from IT3160.evaluation.checkpoint import infer_model_config, load_policy_model
from IT3160.evaluation.inference import (
    Predictions, load_prediction_metadata, load_predictions, model_predict_fn, run_inference,
    save_predictions, score_batch,
)
from IT3160.evaluation.report import slices_table
from IT3160.evaluation.meta import build_meta, load_meta, save_meta
from IT3160.evaluation.metrics import expected_calibration_error, summarize, summarize_by_phase
from IT3160.evaluation.runner import evaluate_predictions, paired_comparison
from IT3160.evaluation.significance import bootstrap_mean_ci, mcnemar_test
from IT3160.evaluation.slices import all_slices, elo_bins, legal_moves_bins, move_features
from IT3160.models.resnet import ChessPolicyResNet

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return ChessPolicyResNet(num_blocks=2, channels=16).eval()


def test_score_batch_matches_masked_softmax():
    torch.manual_seed(0)
    logits = torch.randn(3, 4672)
    mask = torch.zeros(3, 4672, dtype=torch.bool)
    mask[:, :10] = True
    mask[2, :] = False
    mask[2, [7, 9]] = True
    y = torch.tensor([3, 0, 9])
    stats = score_batch(logits, y, mask)

    expected = F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    assert torch.allclose(stats["logp_y"], expected.gather(1, y[:, None]).squeeze(1))
    for i in range(3):
        legal = expected[i][mask[i]]
        assert stats["rank_y"][i] == int((legal >= expected[i, y[i]]).sum())
        assert math.isclose(float(stats["entropy"][i]), float(-(legal.exp() * legal).sum()), rel_tol=1e-5)
    assert stats["n_legal"].tolist() == [10, 10, 2]
    assert stats["top_ids"][2, 0].item() in (7, 9)


def test_rank_ties_count_against_the_label():
    logits = torch.zeros(2, 4672)
    logits[1, 2] = 1.0  # row 1: label 0 tied with action 1, action 2 strictly higher
    mask = torch.zeros(2, 4672, dtype=torch.bool)
    mask[:, :3] = True
    stats = score_batch(logits, torch.tensor([0, 0]), mask)
    assert stats["rank_y"].tolist() == [3, 3]  # all-tie row is not a Top-1 hit


def test_slices_report_random_baseline():
    n_legal = np.array([2, 4, 20, 30], np.int16)
    predictions = summary_predictions(n_legal)
    slices = all_slices(predictions, None)["legal_moves"]
    assert slices["<=10"]["random_top1"] == pytest.approx((1 / 2 + 1 / 4) / 2 * 100)
    assert slices["<=10"]["random_nll"] == pytest.approx((math.log(2) + math.log(4)) / 2)
    assert "Random Top-1" in slices_table({"legal_moves": slices})


def summary_predictions(n_legal):
    n = len(n_legal)
    return Predictions(
        row_idx=np.arange(n), y=np.zeros(n, np.int64), phase=np.zeros(n, np.uint8),
        n_legal=n_legal, logp_y=np.full(n, -1.0, np.float32), rank_y=np.ones(n, np.int32),
        top_ids=np.zeros((n, 5), np.int32), top1_logp=np.full(n, -1.0, np.float32),
        entropy=np.ones(n, np.float32), unmasked_top1_legal=np.ones(n, bool),
    )


def test_inference_matches_model_and_roundtrips(exported_split, tmp_path):
    h5_path, _, records = exported_split
    model = tiny_model()
    predictions = run_inference(model_predict_fn(model, torch.device("cpu")), h5_path,
                                batch_size=5, progress=False)
    assert len(predictions) == len(records)
    assert predictions.y.tolist() == [r.action_id for r in records]
    assert predictions.n_legal.tolist() == [len(r.legal_action_ids) for r in records]

    # Reference: the model's own masking path used during training.
    from IT3160.data.canonical import encode_board
    x = torch.stack([torch.from_numpy(encode_board(chess.Board(r.fen))).float() for r in records])
    mask = torch.zeros(len(records), 4672, dtype=torch.bool)
    for i, r in enumerate(records):
        mask[i, list(r.legal_action_ids)] = True
    y = torch.tensor([r.action_id for r in records])
    with torch.no_grad():
        nll = F.cross_entropy(model(x, legal_mask=mask), y).item()
    assert math.isclose(summarize(predictions)["nll"], nll, rel_tol=1e-4)

    path = tmp_path / "pred.parquet"
    save_predictions(predictions, path, metadata={"policy": "test"})
    loaded = load_predictions(path)
    assert np.array_equal(loaded.top_ids, predictions.top_ids)
    assert np.array_equal(loaded.rank_y, predictions.rank_y)
    assert load_prediction_metadata(path) == {"policy": "test"}


def test_phase_nll_is_per_sample():
    from IT3160.evaluation.inference import Predictions
    n = 4
    predictions = Predictions(
        row_idx=np.arange(n), y=np.zeros(n, np.int64), phase=np.array([0, 0, 1, 1], np.uint8),
        n_legal=np.full(n, 20, np.int16), logp_y=np.log(np.array([0.5, 0.5, 0.1, 0.1], np.float32)),
        rank_y=np.array([1, 1, 2, 3], np.int32), top_ids=np.zeros((n, 5), np.int32),
        top1_logp=np.log(np.array([0.5, 0.5, 0.3, 0.3], np.float32)),
        entropy=np.ones(n, np.float32), unmasked_top1_legal=np.array([True, True, True, False]),
    )
    result = summarize_by_phase(predictions)
    assert math.isclose(result["opening"]["nll"], -math.log(0.5), rel_tol=1e-6)
    assert math.isclose(result["middlegame"]["nll"], -math.log(0.1), rel_tol=1e-6)
    assert result["opening"]["top1"] == 100.0 and result["middlegame"]["top1"] == 0.0
    assert result["middlegame"]["top3"] == 100.0
    assert result["overall"]["unmasked_illegal_top1"] == 25.0
    assert result["endgame"] == {"samples": 0}


def test_random_baseline_is_exact(exported_split):
    h5_path, _, records = exported_split
    n = np.array([len(r.legal_action_ids) for r in records], float)
    result = random_legal_summary(h5_path)["overall"]
    assert math.isclose(result["nll"], np.log(n).mean())
    assert math.isclose(result["top1"], (1 / n).mean() * 100)
    assert math.isclose(result["top5"], (np.minimum(5, n) / n).mean() * 100)


def test_prior_baseline_prefers_frequent_actions(exported_split):
    h5_path, _, _ = exported_split
    prior = fit_action_prior(h5_path)
    assert prior.shape == (3, 4672)
    assert np.allclose(np.exp(prior).sum(axis=1), 1.0, atol=1e-4)
    predictions = run_inference(prior_predict_fn(prior), h5_path, progress=False)
    random = random_legal_summary(h5_path)["overall"]
    # Fitted on the same rows, the prior must beat uniform guessing.
    assert summarize(predictions)["nll"] < random["nll"]


def test_checkpoint_roundtrip_infers_architecture(tmp_path):
    model = tiny_model()
    assert infer_model_config(model.state_dict()) == {
        "in_channels": 17, "channels": 16, "num_blocks": 2, "policy_planes": 73,
    }
    path = tmp_path / "best_model.pt"
    torch.save({"epoch": 3, "model_state_dict": model.state_dict(), "best_val_loss": 1.5,
                "config": {"lr": 3e-4}, "history": []}, path)
    loaded, info = load_policy_model(path)
    assert info["epoch"] == 3 and info["model_config"]["num_blocks"] == 2
    x = torch.randn(2, 17, 8, 8)
    with torch.no_grad():
        assert torch.equal(loaded(x), model(x))


def test_meta_is_aligned_and_detects_mismatch(exported_split, tmp_path):
    h5_path, processed_dir, records = exported_split
    table = build_meta(h5_path, processed_dir=processed_dir)
    assert table.num_rows == len(records)
    save_meta(table, tmp_path / "meta.parquet")
    meta = load_meta(tmp_path / "meta.parquet")
    assert list(meta["move_uci"]) == [r.move_uci for r in records]

    import h5py
    with h5py.File(h5_path, "r+") as f:
        f["y"][0] = (f["y"][0] + 1) % 4672
    y = np.array([r.action_id for r in records])
    meta = load_meta(tmp_path / "meta.parquet", np.arange(len(y)), expected_y=y, total_rows=len(y))
    assert len(meta["fen"]) == len(y)
    with pytest.raises(ValueError, match="different split"):  # meta of another split
        load_meta(tmp_path / "meta.parquet", np.arange(len(y)), expected_y=np.roll(y, 1))
    with pytest.raises(ValueError, match="rows but the evaluated split"):
        load_meta(tmp_path / "meta.parquet", total_rows=len(y) + 1)

    with pytest.raises(ValueError, match="checksum|does not match"):
        build_meta(h5_path, processed_dir=processed_dir)


def test_slices_and_features(exported_split):
    _, _, records = exported_split
    fens = np.array([r.fen for r in records], object)
    moves = np.array([r.move_uci for r in records], object)
    features = move_features(fens, moves)
    kinds = dict(zip(moves, features["move_kind"]))
    assert kinds["e1g1"] == "castling"
    assert kinds["b7a8q"] == "promotion"
    assert kinds["f3g5"] == "capture"
    assert kinds["e2e4"] == "quiet"
    assert list(legal_moves_bins(np.array([1, 10, 11, 40, 41]))) == ["<=10", "<=10", "11-25", "26-40", ">40"]
    assert list(elo_bins(np.array([2300.0, np.nan]), np.array([2300.0, 2500.0]))) == ["<2400", "unknown"]


def test_bootstrap_and_mcnemar():
    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 1.0, size=2000)
    ci = bootstrap_mean_ci(values, n_boot=300, seed=1)
    assert ci["low"] < 5.0 < ci["high"] and ci["low"] < ci["estimate"] < ci["high"]
    grouped = bootstrap_mean_ci(values, np.repeat(np.arange(200), 10), n_boot=300, seed=1)
    assert grouped["estimate"] == ci["estimate"]

    a = np.array([1] * 10 + [0] * 50, bool)
    b = np.array([0] * 10 + [1] * 50, bool)
    exact = mcnemar_test(a, b)  # 10 vs 50 discordant
    assert exact["only_a_correct"] == 10 and exact["only_b_correct"] == 50
    assert exact["method"] == "exact" and exact["p_value"] < 1e-5
    assert mcnemar_test(a, a)["p_value"] == 1.0
    big_a = np.r_[np.ones(700, bool), np.zeros(700, bool)]
    assert mcnemar_test(big_a, ~big_a)["method"] == "chi2_cc"


def test_ece_perfect_and_overconfident():
    assert expected_calibration_error(np.full(100, 0.95), np.ones(100, bool)) == pytest.approx(5.0 / 100)
    assert expected_calibration_error(np.full(4, 1.0), np.array([1, 1, 0, 0], bool)) == pytest.approx(0.5)


def test_evaluate_and_paired_comparison(exported_split):
    h5_path, processed_dir, _ = exported_split
    meta = load_meta_from(exported_split)
    a = run_inference(model_predict_fn(tiny_model(0), torch.device("cpu")), h5_path, progress=False)
    b = run_inference(model_predict_fn(tiny_model(1), torch.device("cpu")), h5_path, progress=False)
    result = evaluate_predictions(a, meta, n_boot=50)
    assert result["ci_resampling"] == "game"
    assert set(result["slices"]) >= {"legal_moves", "move_kind", "piece", "ply", "elo", "gives_check"}
    assert result["worst_examples"]
    json.dumps(result)  # serialisable
    rows = paired_comparison("a", a, "b", b, meta, n_boot=50)
    overall = next(r for r in rows if r["group"] == "overall")
    expected = ((b.rank_y == 1).mean() - (a.rank_y == 1).mean()) * 100
    assert overall["delta_top1"]["estimate"] == pytest.approx(expected)


def load_meta_from(exported_split):
    h5_path, processed_dir, _ = exported_split
    table = build_meta(h5_path, processed_dir=processed_dir)
    return {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}


def test_scripts_end_to_end(exported_split, tmp_path, monkeypatch):
    h5_path, processed_dir, _ = exported_split
    monkeypatch.syspath_prepend(str(SCRIPTS))
    for name in ("evaluate", "compare", "export_meta"):
        sys.modules.pop(name, None)
    import compare
    import evaluate
    import export_meta

    meta_path = tmp_path / "test_meta.parquet"
    assert export_meta.main(["--h5", str(h5_path), "--processed-dir", str(processed_dir),
                             "--out", str(meta_path)]) == 0
    for seed in (0, 1):
        checkpoint = tmp_path / f"m{seed}" / "best_model.pt"
        checkpoint.parent.mkdir()
        torch.save({"epoch": 1, "model_state_dict": tiny_model(seed).state_dict()}, checkpoint)
        assert evaluate.main(["--checkpoint", str(checkpoint), "--data", str(h5_path),
                              "--meta", str(meta_path), "--bootstrap", "20", "--device", "cpu"]) == 0
    out0 = tmp_path / "m0" / "eval" / "best_model__test"
    metrics = json.loads((out0 / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["overall"]["samples"] > 0 and "slices" in metrics
    assert (out0 / "metrics.md").is_file()

    assert evaluate.main(["--baseline", "random", "--data", str(h5_path),
                          "--out-dir", str(tmp_path / "random")]) == 0
    assert evaluate.main(["--baseline", "prior", "--train-path", str(h5_path), "--data", str(h5_path),
                          "--out-dir", str(tmp_path / "prior"), "--bootstrap", "0"]) == 0

    assert compare.main(["--runs", str(out0), str(tmp_path / "m1" / "eval" / "best_model__test"),
                         str(tmp_path / "prior"), "--names", "A", "B", "Prior",
                         "--meta", str(meta_path), "--bootstrap", "20",
                         "--out", str(tmp_path / "cmp")]) == 0
    comparison = json.loads((tmp_path / "cmp" / "comparison.json").read_text(encoding="utf-8"))
    assert len(comparison["paired_vs_reference"]) == 2 * 4
