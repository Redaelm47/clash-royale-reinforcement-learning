"""Head to head between two checkpoints, with perfect input or "real game" input (--real-game: the PC
kit noise and the 0.25 s delay, on both sides). 400 games take about 3 min on a Mac.

--temp-a / --temp-b: temperature of each side, in "inner" mode like the kit (when to play is still
sampled as is, only the card and the tile are sharpened). The kit plays at 0.5. Default 1 (training).
The cards played per minute by each side are also printed: a high score against an opponent
that barely plays means nothing.

Usage:
  .venv/bin/python eval/head_to_head.py runs/play/reel_it0400.pt runs/play/principal_it1450_97pct.pt --games 400 --real-game \
      --temp-a 0.5 --temp-b 0.5
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.play import model_policy, run_games
from models.policy import load, HEADER, ENT
from crforge_env import LadderConfig, VectorBattle

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--games", type=int, default=200)
ap.add_argument("--real-game", action="store_true")
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--envs", type=int, default=40)
ap.add_argument("--temp-a", type=float, default=1.0)
ap.add_argument("--temp-b", type=float, default=1.0)
args = ap.parse_args()
v = VectorBattle(args.envs, 8, install=str(ROOT / "sim/crforge/env/build/install/env"),
                 tables=str(Path.home() / ".crforge/tables/16.635.8"), config=LadderConfig())
s = v.schema
ci = s.header["entity_count"][0]
E_KIND, E_X, E_Y, E_DX, E_DY = (s.entity[k] for k in ("kind", "x", "y", "dx", "dy"))
play_fields = [s.header[k] for k in ("enemy_play_card", "enemy_play_cast", "enemy_play_form", "enemy_play_ticks",
                                     "enemy_play_x", "enemy_play_y", "enemy_play_kind", "enemy_play_flags",
                                     "enemy_play_point_ticks")]
h_el = s.header["enemy_elixir"][0]
rng = np.random.default_rng(args.seed)
p_drop, sigma, p_play = 0.2, 0.3, 0.15


def corrupt(ob):
    ob = ob.copy(); n = len(ob)
    n_ent = (ob.shape[1] - HEADER) // ENT
    ent = ob[:, HEADER: HEADER + n_ent * ENT].reshape(n, n_ent, ENT)
    count = ob[:, ci].clip(0, n_ent)
    idx = np.arange(n_ent)[None, :]
    unit = (idx >= 6) & (idx < count[:, None]) & (ent[..., E_KIND] == 1)
    drop = unit & (rng.random((n, n_ent)) < p_drop)
    move = unit & ~drop
    ent[..., E_X] = np.where(move, (ent[..., E_X] + rng.normal(0, sigma * 1000, (n, n_ent))).clip(0, 18000), ent[..., E_X])
    ent[..., E_Y] = np.where(move, (ent[..., E_Y] + rng.normal(0, sigma * 1000, (n, n_ent))).clip(0, 32000), ent[..., E_Y])
    for f in (E_DX, E_DY):
        ent[..., f] = np.where(move, ent[..., f] + rng.normal(0, sigma * 300, (n, n_ent)), ent[..., f])
    if drop.any():
        order = np.argsort(drop, axis=1, kind="stable")
        ent[:] = np.take_along_axis(ent, order[..., None], axis=1)
        new_count = count - drop.sum(1)
        ent[idx >= new_count[:, None]] = 0
        ob[:, ci] = new_count
    ob[:, HEADER: HEADER + n_ent * ENT] = ent.reshape(n, -1)
    rows = len(play_fields[0])
    shown = ob[:, play_fields[0]] >= 0
    miss = shown & (rng.random((n, rows)) < p_play)
    order = np.argsort(miss, axis=1, kind="stable")
    kept = (~miss).sum(1)
    blank = np.arange(rows)[None, :] >= kept[:, None]
    for j, cols in enumerate(play_fields):
        cols = np.array(cols)
        vals = np.take_along_axis(ob[:, cols], order, axis=1)
        vals = np.where(blank, -1 if j in (0, 1) else 0, vals)
        live = ~blank & (vals != 0)
        if j == 3:
            vals = np.where(live, (vals + rng.integers(-6, 7, vals.shape)).clip(1), vals)
        elif j in (4, 5):
            vals = np.where(live & (vals > 0), vals + rng.normal(0, 600, vals.shape).astype(np.int64), vals)
        ob[:, cols] = vals
    ob[:, h_el] = (ob[:, h_el] + rng.normal(0, 7000, n)).clip(0, 100000)
    return ob


def wrap(pol, count):
    prev = np.zeros(args.envs, dtype=np.int64)
    def act(obs, mask):
        if args.real_game:
            a = pol(corrupt(obs), mask)
            go = np.where(mask[np.arange(len(obs)), prev] != 0, prev, 0)
            prev[:] = a
        else:
            go = pol(obs, mask)
        count[0] += int((go != 0).sum())
        count[1] += len(go)
        return go
    return act


ma, _ = load(args.a, "cpu"); mb, _ = load(args.b, "cpu"); ma.eval(); mb.eval()
t = time.time()
ca, cb = [0, 0], [0, 0]  # plays made, decisions
r = run_games(v, wrap(model_policy(ma, "cpu", ci, args.temp_a), ca), wrap(model_policy(mb, "cpu", ci, args.temp_b), cb),
              args.games, seed=args.seed)
n = r["games"]
per_min = lambda c: 240 * c[0] / max(1, c[1])  # 240 decisions per minute
print(f"{Path(args.a).stem} (T {args.temp_a:g}) vs {Path(args.b).stem} (T {args.temp_b:g}) "
      f"({'real game' if args.real_game else 'perfect'}): {r['score']:.1%} over {n} games "
      f"(W {r['wins']} L {r['losses']} D {r['draws']}), ±{1.96 * np.sqrt(0.25 / n):.1%} | cards per minute "
      f"{per_min(ca):.1f} vs {per_min(cb):.1f} | {time.time() - t:.0f} s", flush=True)
