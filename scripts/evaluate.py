"""Evaluate a checkpoint (or a baseline) on one HDF5 split.

Writes to --out-dir:
  predictions.parquet  per-sample results (input of scripts/compare.py)
  metrics.json         overall / per-phase metrics, bootstrap CIs, calibration, slices
  metrics.md           the same as Markdown tables

Examples:
  python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
  python scripts/evaluate.py --checkpoint runs/e1/best_model.pt \
      --data data/tensors/official-v1/test.h5 --meta data/tensors/official-v1/test_meta.parquet
  python scripts/evaluate.py --baseline prior --train-path data/tensors/official-v1/train.h5
  python scripts/evaluate.py --baseline random
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from IT3160.evaluation.baselines import (  # noqa: E402
    fit_action_prior, prior_predict_fn, random_legal_summary,
)
from IT3160.evaluation.checkpoint import load_policy_model  # noqa: E402
from IT3160.evaluation.inference import (  # noqa: E402
    h5_length, model_predict_fn, run_inference, save_predictions,
)
from IT3160.evaluation.meta import load_meta  # noqa: E402
from IT3160.evaluation.report import metrics_table, slices_table  # noqa: E402
from IT3160.evaluation.runner import evaluate_predictions  # noqa: E402

DEFAULT_DATA = "data/tensors/official-v1/test.h5"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Model checkpoint (.pt) written by scripts/train.py")
    source.add_argument("--baseline", choices=("random", "prior"), help="Evaluate a reference policy")
    parser.add_argument("--data", "--test-path", dest="data", default=DEFAULT_DATA,
                        help=f"HDF5 split to evaluate (default: {DEFAULT_DATA})")
    parser.add_argument("--meta", help="Row metadata parquet (scripts/export_meta.py); enables "
                        "move/ply/Elo slices and game-level bootstrap")
    parser.add_argument("--train-path", default="data/tensors/official-v1/train.h5",
                        help="Training split used to fit the prior baseline")
    parser.add_argument("--out-dir", help="Output folder (default: next to the checkpoint)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples (0 = off)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--amp", action="store_true",
                        help="fp16 autocast on CUDA (faster; default fp32 so CPU/GPU results match)")
    return parser.parse_args(argv)


def repo_relative(path: str | Path) -> str:
    """Path relative to the repo when inside it, so saved results carry no machine-specific prefix."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def default_out_dir(args) -> Path:
    data = Path(args.data).stem
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        return checkpoint.parent / "eval" / f"{checkpoint.stem}__{data}"
    return REPO_ROOT / "results" / "baselines" / f"{args.baseline}__{data}"


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"data": repo_relative(args.data), "limit": args.limit, "device": str(device)}

    if args.baseline == "random":
        metrics = random_legal_summary(args.data, limit=args.limit)
        info["policy"] = "random-legal (analytic expectation)"
        if args.meta:
            print("Note: --meta is ignored for --baseline random; per-slice random columns are "
                  "reported in every model's slice tables instead.")
    else:
        if args.checkpoint:
            model, model_info = load_policy_model(args.checkpoint, device)
            use_amp = args.amp and device.type == "cuda"
            predict = model_predict_fn(model, device, use_amp=use_amp)
            info.update(policy="model", amp=use_amp, **model_info)
            info["checkpoint"] = repo_relative(args.checkpoint)
            print(f"Model: {model_info['model_config']} | {model_info['parameters']:,} params"
                  f" | epoch {model_info['epoch']}")
        else:
            predict = prior_predict_fn(fit_action_prior(args.train_path), device)
            info.update(policy="phase-conditional action prior", train_path=repo_relative(args.train_path))
        print(f"Evaluating {h5_length(args.data, args.limit):,} rows of {args.data} on {device}")
        predictions = run_inference(predict, args.data, device=device,
                                    batch_size=args.batch_size, limit=args.limit)
        meta = load_meta(args.meta, predictions.row_idx, expected_y=predictions.y,
                         total_rows=h5_length(args.data)) if args.meta else None
        metrics = evaluate_predictions(predictions, meta, n_boot=args.bootstrap, seed=args.seed)
        save_predictions(predictions, out_dir / "predictions.parquet", metadata=info)

    metrics["info"] = info
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    markdown = f"# Evaluation\n\n- policy: {info['policy']}\n- data: {info['data']}\n\n"
    markdown += metrics_table(metrics) + "\n"
    if metrics.get("slices"):
        markdown += "\n## Slices\n" + slices_table(metrics["slices"]) + "\n"
    (out_dir / "metrics.md").write_text(markdown, encoding="utf-8")
    print(metrics_table(metrics))
    print(f"\nSaved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
