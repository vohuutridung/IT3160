"""Play a controlled match between two players and save W/D/L, Elo estimate and PGN.

A player is a checkpoint path (.pt) or one of: random, greedy.
Every opening is played twice with colours swapped; openings are distinct
positions sampled from the TEST games (plies --min-ply..--max-ply) so that no
model has trained on them.

Examples:
  python scripts/play_match.py --a runs/e1/best_model.pt --b random --openings 25
  python scripts/play_match.py --a runs/e1/best_model.pt --b runs/e2/best_model.pt \
      --names E1 E2 --openings 50 --out results/matches/e1_vs_e2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from IT3160.evaluation.agent import PolicyAgent  # noqa: E402
from IT3160.evaluation.arena import (  # noqa: E402
    GreedyMaterialPlayer, ModelPlayer, RandomPlayer, game_to_dict, openings_from_meta,
    play_match, summarize_match, to_pgn,
)
from IT3160.evaluation.checkpoint import load_policy_model  # noqa: E402
from IT3160.evaluation.meta import load_meta  # noqa: E402

DEFAULT_META = "data/tensors/official-v1/test_meta.parquet"


def make_player(spec: str, name: str, device: torch.device):
    if spec == "random":
        return RandomPlayer(name)
    if spec == "greedy":
        return GreedyMaterialPlayer(name)
    model, _ = load_policy_model(spec, device)
    return ModelPlayer(name, PolicyAgent(model, device))


def default_name(spec: str) -> str:
    path = Path(spec)
    return spec if spec in ("random", "greedy") else f"{path.parent.name}/{path.stem}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="Player A: checkpoint .pt | random | greedy")
    parser.add_argument("--b", required=True, help="Player B: checkpoint .pt | random | greedy")
    parser.add_argument("--names", nargs=2, help="Display names for A and B")
    parser.add_argument("--openings", type=int, default=50, help="Number of openings (games = 2x)")
    parser.add_argument("--openings-file", help="Text file with one FEN per line (overrides --meta)")
    parser.add_argument("--meta", default=DEFAULT_META, help="Row metadata to draw openings from")
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=12)
    parser.add_argument("--max-plies", type=int, default=300, help="Adjudicate a draw after N plies")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = always the most likely move; >0 samples p^(1/T)")
    parser.add_argument("--allow-repetition", action="store_true",
                        help="Do not skip moves that allow a threefold-repetition claim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", help="Output folder (default: results/matches/<a>_vs_<b>)")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    names = args.names or [default_name(args.a), default_name(args.b)]
    if names[0] == names[1]:
        parser.error("Players need distinct names; use --names")
    player_a = make_player(args.a, names[0], device)
    player_b = make_player(args.b, names[1], device)

    if args.openings_file:
        lines = Path(args.openings_file).read_text(encoding="utf-8").splitlines()
        openings = [line.strip() for line in lines if line.strip()][: args.openings]
    else:
        meta = load_meta(args.meta)
        openings = openings_from_meta(meta["fen"], meta["ply"], args.openings,
                                      min_ply=args.min_ply, max_ply=args.max_ply, seed=args.seed)

    safe = "_vs_".join(n.replace("/", "-") for n in names)
    out = Path(args.out) if args.out else REPO_ROOT / "results" / "matches" / safe
    out.mkdir(parents=True, exist_ok=True)
    print(f"{names[0]} vs {names[1]}: {len(openings)} openings x 2 colours on {device}")

    games = play_match(player_a, player_b, openings, seed=args.seed, max_plies=args.max_plies,
                       temperature=args.temperature, avoid_repetition=not args.allow_repetition)
    summary = summarize_match(games, names[0], names[1])
    summary["settings"] = {k: v for k, v in vars(args).items() if k != "names"} | {"names": names}

    (out / "openings.txt").write_text("\n".join(openings) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "games.jsonl").open("w", encoding="utf-8") as handle:
        for game in games:
            handle.write(json.dumps(game_to_dict(game)) + "\n")
    (out / "games.pgn").write_text(
        "\n\n".join(to_pgn(g, round_number=i + 1) for i, g in enumerate(games)) + "\n",
        encoding="utf-8",
    )

    s = summary
    print(f"\n{names[0]}: +{s['a_wins']} ={s['draws']} -{s['a_losses']}  "
          f"score {s['a_score']:.3f} [{s['a_score_ci95'][0]:.3f}, {s['a_score_ci95'][1]:.3f}]")
    if s["elo_diff_ci95"]:
        print(f"Elo diff ({names[0]} - {names[1]}): {s['elo_diff']:+.0f} "
              f"[{s['elo_diff_ci95'][0]:+.0f}, {s['elo_diff_ci95'][1]:+.0f}]")
    print(f"As White {s['a_as_white']['score']:.3f} | as Black {s['a_as_black']['score']:.3f} | "
          f"mean plies {s['mean_plies']:.0f}")
    print(f"Terminations: {s['terminations']}")
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
