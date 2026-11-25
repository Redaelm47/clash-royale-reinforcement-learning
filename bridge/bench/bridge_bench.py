"""Bench: what the PC bridge costs the bot, measured in crforge.

A gets the input "as on the PC": imperfect vision, crforge twin, corrections.
B gets the perfect input. Same model on both sides: 50 % means the bridge costs nothing.

  .venv/bin/python bridge/bench/bridge_bench.py --games 20 --variants 4 --miss-rate 0.2 --noise 0.3 --missed-plays 0.15
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "bridge" / "pc"
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--games", type=int, default=20)
ap.add_argument("--variants", type=int, default=4)
ap.add_argument("--miss-rate", type=float, default=0.2, help="share of units the vision misses on each frame")
ap.add_argument("--noise", type=float, default=0.3, help="standard deviation of the seen positions (tiles)")
ap.add_argument("--missed-plays", type=float, default=0.15, help="share of enemy plays that are missed")
ap.add_argument("--background", action="store_true", help="twin rebuilds in the background")
ap.add_argument("--model", default=str(ROOT / "runs/play/principal_it1450_97pct.pt"))
ap.add_argument("--seed", type=int, default=5000)
ap.add_argument("--perfect", action="store_true", help="control: A also gets the perfect input (should give ~50 %%)")
args = ap.parse_args()
import twin
from state import Encoder
_s = twin.settings
def fake():
    d = _s(); d.update(my_card_level=16, my_tower_level=16, enemy_card_level=16, enemy_tower_level=16,
                       my_evolutions=["Cannon", "Skeletons"], my_heroes=["Musketeer"], enemy_evolutions=["Cannon", "Skeletons"], enemy_heroes=["Musketeer"])
    return d
twin.settings = fake
tw = twin.Twin(n=args.variants, background=args.background)
from crforge_lite import LadderConfig, SideSetup, VectorBattle
import torch
from models.policy import load
from eval.play import model_policy
from sim.cards import DECK_26, SLOTS_26
meta = json.load(open(KIT / "model/metadata.json")); enc = Encoder(meta); names, H, E = enc.names, enc.H, enc.E
real = VectorBattle(1, 1, install=str(twin.INSTALL), tables=str(twin.TABLES), config=LadderConfig(arena="Arena_competitive"))
model, _ = load(args.model, "cpu"); model.eval()
pol = model_policy(model, "cpu", H["entity_count"][0], 1.0)
setup = SideSetup(king_level=16, tower_level=16, card_levels=[16] * 8)
HI, EI = 224, 34
def units_of(o):
    out = []
    for k in range(6, int(o[H["entity_count"][0]])):
        r = o[HI + k * EI: HI + (k + 1) * EI]
        if r[E["kind"]] == 1:
            out.append({"row": names[r[E["row"]]], "side": int(r[E["side"]]), "x": r[E["x"]] / 1000, "y": r[E["y"]] / 1000,
                        "seen": 3, "max": int(r[E["max_hp"]]), "rowid": int(r[E["row"]])})
    return out
def screen_obs(o, vision):
    so = o.copy(); so[HI + 6 * EI:] = 0
    for j, u in enumerate(vision[:150]):
        a = HI + (6 + j) * EI
        so[a + E["id"]], so[a + E["kind"]], so[a + E["side"]], so[a + E["row"]] = 100 + j, 1, u["side"], u["rowid"]
        so[a + E["x"]], so[a + E["y"]] = int(u["x"] * 1000), int(u["y"] * 1000)
        so[a + E["hp"]], so[a + E["max_hp"]], so[a + E["level"]] = u["max"], u["max"], 16
    so[H["entity_count"][0]] = 6 + min(150, len(vision))
    return so
res = {"A": 0, "B": 0, "draw": 0}; t_start = time.time()
for g in range(args.games):
    random.seed(g); torch.manual_seed(g); np.random.seed(g)
    seat_a = g % 2
    real.reset([args.seed + g], DECK_26, DECK_26, SLOTS_26, SLOTS_26, setup, setup)
    tw.reset(); t0 = 1000.0; pending = []
    for step in range(1200):
        tick = step * 5; now = t0 + tick / 20
        o, m = real.observations[0], real.masks[0]
        a_b = pol(o[1 - seat_a][None], m[1 - seat_a][None])[0]
        oa = o[seat_a]
        if tick >= 100 and not args.perfect:
            for item in [p for p in pending if p[0] <= now]:
                pending.remove(item)
                _, card, t_run, x, y = item
                if random.random() > args.missed_plays:
                    tw.add_enemy(card, t_run + random.uniform(-0.3, 0.3) * (args.noise > 0), x + random.uniform(-0.6, 0.6) * (args.noise > 0),
                                 y + random.uniform(-0.6, 0.6) * (args.noise > 0))
                elif random.random() < 0.5:
                    tw.add_enemy(None, t_run, x, y)
            truth_u = units_of(oa)
            vision = [dict(u, x=u["x"] + random.gauss(0, args.noise), y=u["y"] + random.gauss(0, args.noise))
                      for u in truth_u if random.random() >= args.miss_rate]
            if tw.sync(now, t0):
                fr = [oa[HI + k * EI + E["hp"]] / max(1, oa[HI + k * EI + E["max_hp"]]) for k in range(6)]
                tw.judge(vision, fr, now)
                so = screen_obs(oa, vision)
                v = twin.view(tw, tw.observation(), so, enc)
                v, _ = twin.drop_ghosts(v, tw, vision, now, enc)
                oa, _ = twin.add_unseen(v, so, H, E, HI, EI)
        a_a = pol(oa[None], m[seat_a][None])[0]
        a_a = 0 if a_a >= 2305 else a_a
        a_b = 0 if a_b >= 2305 else a_b
        if a_a > 0:
            slot, k = (a_a - 1) // 576, (a_a - 1) % 576
            tw.add_own(names[o[seat_a][H["hand_card"][slot]]], now, (k % 18, k // 18))
        if a_b > 0:
            slot, k = (a_b - 1) // 576, (a_b - 1) % 576
            col, row = k % 18, k // 18
            pending.append((now + 1.3, names[o[1 - seat_a][H["hand_card"][slot]]], now + 1.0, 18 - (col + 0.5), 32 - (row + 0.5)))
        real.actions[0, seat_a], real.actions[0, 1 - seat_a] = a_a, a_b
        real.step()
        if real.dones[0]:
            break
    w = int(real.infos[0][2])
    res["A" if w == seat_a else ("B" if w == 1 - seat_a else "draw")] += 1
    print(f"game {g + 1}/{args.games}: A {res['A']} B {res['B']} draws {res['draw']} ({time.time() - t_start:.0f} s)", flush=True)
score = (res["A"] + 0.5 * res["draw"]) / args.games
print(f"RESULT {'CONTROL perfect/perfect, ' if args.perfect else ''}{args.variants} variant(s){', background' if args.background else ''}, missed units {args.miss_rate:.0%}, noise {args.noise}, "
      f"missed plays {args.missed_plays:.0%}: A (PC input) scores {score:.0%} against B (perfect input), {res}")
