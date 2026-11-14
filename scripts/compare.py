"""Compare evaluated models on the same split (paired tests against the first one).

Each --runs entry is an evaluation folder containing predictions.parquet
(written by scripts/evaluate.py). The first run is the reference.

Example:
  python scripts/compare.py --runs runs/e1/eval/best_model__test runs/e2/eval/best_model__test \
      --names E1-balanced E2-natural --meta data/tensors/official-v1/test_meta.parquet \
      --out results/compare_e1_e2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from IT3160.evaluation.inference import load_predictions  # noqa: E402
from IT3160.evaluation.meta import load_meta  # noqa: E402
from IT3160.evaluation.metrics import GROUPS, summarize_by_phase  # noqa: E402
from IT3160.evaluation.report import comparison_table, paired_table  # noqa: E402
from IT3160.evaluation.runner import check_aligned, paired_comparison  # noqa: E402


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # tables contain "Δ"; Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="Evaluation folders; first = reference")
    parser.add_argument("--names", nargs="+", help="Display names (default: folder names)")
    parser.add_argument("--meta", help="Row metadata parquet for game-level bootstrap")
    parser.add_argument("--out", default="results/compare", help="Output folder")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    runs = [Path(r) for r in args.runs]
    names = args.names or [r.name if r.name else str(r) for r in runs]
    if len(names) != len(runs):
        parser.error("--names must match --runs")
    predictions = [load_predictions(r / "predictions.parquet") for r in runs]
    for other in predictions[1:]:
        check_aligned(predictions[0], other)
    meta = (load_meta(args.meta, predictions[0].row_idx, expected_y=predictions[0].y)
            if args.meta else None)

    summaries = [summarize_by_phase(p) for p in predictions]
    paired = []
    for name, other in zip(names[1:], predictions[1:]):
        paired += paired_comparison(names[0], predictions[0], name, other, meta,
                                    n_boot=args.bootstrap, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps({
        "runs": dict(zip(names, map(str, runs))),
        "summaries": dict(zip(names, summaries)),
        "paired_vs_reference": paired,
        "ci_resampling": "game" if meta is not None else "sample",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    sections = [f"## {group}\n\n{comparison_table(names, summaries, group)}" for group in GROUPS]
    if paired:
        sections.append("## Paired tests vs reference\n\n" + paired_table(paired))
    markdown = "# Model comparison\n\n" + "\n\n".join(sections) + "\n"
    (out / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
