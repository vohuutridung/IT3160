"""Write <split>_meta.parquet: FEN, move, ply, Elo and game ID for every HDF5 row.

Source shards are the ones recorded in the HDF5 metadata, downloaded from the
exact Hugging Face revision used for export (or read from --processed-dir),
checksum-verified, and matched row by row against the HDF5 labels.

Example:
  python scripts/export_meta.py --h5 data/tensors/official-v1/test.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from IT3160.evaluation.meta import build_meta, save_meta  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", required=True, help="Exported split, e.g. .../test.h5")
    parser.add_argument("--out", help="Output parquet (default: <h5 folder>/<split>_meta.parquet)")
    parser.add_argument("--processed-dir", help="Local processed shards instead of downloading from HF")
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "data" / "meta_cache"))
    args = parser.parse_args(argv)

    h5 = Path(args.h5)
    out = Path(args.out) if args.out else h5.with_name(f"{h5.stem}_meta.parquet")
    table = build_meta(h5, processed_dir=args.processed_dir, cache_dir=args.cache_dir)
    save_meta(table, out)
    print(f"Wrote {table.num_rows:,} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
