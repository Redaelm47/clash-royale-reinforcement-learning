"""Elo ranking of checkpoints: each pair plays some games, then the ratings are fitted.

Unlike the win rate against a fixed bot, this ranking does not hit a ceiling: each version is
measured against the previous ones. The first player in the list is the zero point (1000 Elo),
usually the imitation bot. For each version we also print its worst score against older
versions: if it goes down, the bot is forgetting.

Usage:
  python eval/league.py --players runs/bc/bc_best.pt "runs/ppo/ppo_it*.pt" runs/ppo/ppo_last.pt scripted \
      --games 64 --install sim/crforge/env/build/install/env --tables ~/.crforge/tables/16.635.8
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.play import make_opponent, model_policy, run_games  # noqa: E402
from models.policy import load  # noqa: E402

from crforge_env import LadderConfig, VectorBattle  # noqa: E402


def fit_elo(names, results, anchor=0, iters=2000, lr=10.0, prior=1.0):
    """Elo ratings that best explain the scores (win 1, draw 0.5), with player `anchor` at 1000.

    `prior` adds a fake drawn game to each pair, to keep the ratings finite when a pair ends
    at 100% or 0%.
    """
    k = len(names)
    r = np.zeros(k)
    for _ in range(iters):
        grad = np.zeros(k)
        for (i, j), (score, n) in results.items():
            score, n = score + prior / 2, n + prior
            p = 1.0 / (1.0 + 10 ** ((r[j] - r[i]) / 400))
            grad[i] += (score - p * n)
            grad[j] -= (score - p * n)
        r += lr * grad / max(1, len(results))
        r -= r[anchor]
    return r + 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", nargs="+", required=True, help="checkpoints (glob patterns allowed), scripted, random")
    ap.add_argument("--games", type=int, default=64, help="games per pair")
    ap.add_argument("--neighbours", type=int, default=0, help="only play against the N neighbours in the list (0 = all)")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--install", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="JSON file for the results")
    args = ap.parse_args()

    names = []
    for p in args.players:
        found = sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
        names += [f for f in found if f not in names]
    v = VectorBattle(args.envs, args.threads, install=args.install,
                     tables=str(Path(args.tables).expanduser()), config=LadderConfig())
    ci = v.schema.header["entity_count"][0]
    models = {n: (load(n, args.device)[0].eval() if n not in ("scripted", "random") else None) for n in names}

    def policy(n):
        if models[n] is None:
            return make_opponent(n, v, args.device, ci)
        return model_policy(models[n], args.device, ci, args.temperature)

    results = {}
    pairs = [(i, j) for i, j in itertools.combinations(range(len(names)), 2)
             if not args.neighbours or j - i <= args.neighbours]
    for i, j in pairs:
        r = run_games(v, policy(names[i]), policy(names[j]), args.games, seed=1000 * i + j)
        results[(i, j)] = (r["wins"] + 0.5 * (r["draws"] + r["cut"]), r["games"])
        print(f"{Path(names[i]).name} vs {Path(names[j]).name}: {r['score']:.0%} over {r['games']}", flush=True)
    elo = fit_elo(names, results)
    print("\nranking:")
    for k in np.argsort(-elo):
        older = [results[(i, k)] for i in range(k) if (i, k) in results]
        worst = min((1 - s / n) for s, n in older) if older else None
        w = "" if worst is None else f" | worst score vs older versions {worst:.0%}"
        print(f"  {elo[k]:7.0f}  {Path(names[k]).name}{w}")
    if args.out:
        Path(args.out).write_text(json.dumps({"players": names, "elo": elo.tolist(),
                                              "results": {f"{i},{j}": v for (i, j), v in results.items()}}, indent=1))
    v.close()


if __name__ == "__main__":
    main()
