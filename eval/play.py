"""Play a checkpoint against an opponent and count the wins.

Possible opponents:
- another checkpoint (path to a .pt);
- "scripted": the crforge rule-based bot (ScriptedBot);
- "random": a random legal card 3 times out of 10, otherwise it waits.
The bot plays seat 0 in half of the games and seat 1 in the other half. 2.6 deck on both
sides. By default the bot samples its plays like in training (temperature 1). A lower
--temperature only sharpens the card and tile choice, not the pace of play. --save-replays writes each game as a crforge replay.

Usage:
  python eval/play.py --model runs/ppo/ppo_last.pt --opponent scripted --games 200 \
      --install sim/crforge/env/build/install/env --tables ~/.crforge/tables/16.635.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.policy import ENT, HEADER, load, sample  # noqa: E402
from sim.cards import DECK_26, SLOTS_26  # noqa: E402

from crforge_env import LadderConfig, ScriptedBot, SideSetup, VectorBattle  # noqa: E402


def model_policy(model, dev, ci, temperature=1.0, max_entities=200):
    """A function (observations, masks) -> actions for a batch of seats."""
    def act(obs, mask):
        n = int(min(max_entities, max(1, obs[:, ci].max())))
        o = torch.from_numpy(np.ascontiguousarray(obs[:, : HEADER + n * ENT])).to(dev)
        k = torch.from_numpy(mask.astype(bool)).to(dev)
        with torch.no_grad():
            f, t, _ = model(o, k)
            if temperature != 1.0:
                # "inner" mode: keep the wait probability as it is (otherwise a low
                # temperature makes the bot passive), only sharpen the card and the tile
                pw = f[:, :1]
                rest = torch.log_softmax(f[:, 1:] / temperature, -1) + torch.log1p(-pw.exp().clamp(max=1 - 1e-6))
                rest = rest.masked_fill(f[:, 1:] < -1e8, -1e9)
                f = torch.cat([pw, rest], 1)
                t = torch.log_softmax(t / temperature, -1)
            return sample(f, t).cpu().numpy()
    return act


def make_opponent(kind, v, dev, ci, temperature=1.0, seed=0):
    """The opponent: "scripted", "random", a checkpoint path, or a model that is already loaded."""
    rng = np.random.default_rng(seed)
    if kind == "scripted":
        bots = [ScriptedBot().bind(v) for _ in range(v.count)]
        return lambda obs, mask: np.array([bots[i](obs[i], mask[i], rng) for i in range(len(obs))])
    if kind == "random":
        def opp(obs, mask):
            out = np.zeros(len(obs), dtype=np.int64)
            for i in range(len(obs)):
                legal = np.flatnonzero(mask[i][1:]) + 1
                if len(legal) and rng.random() < 0.3:
                    out[i] = rng.choice(legal)
            return out
        return opp
    if isinstance(kind, (str, Path)):
        kind, _ = load(kind, dev)
    kind.eval()
    return model_policy(kind, dev, ci, temperature)


def run_games(v, me, opp, games, seed=0, level=16, plain=False, replays=None):
    """Plays `games` games on the vector v; the bot `me` alternates seats. Returns the counts."""
    s = v.schema
    setup = SideSetup(king_level=level, tower_level=level, card_levels=[level] * 8) if level else None
    slots = None if plain else SLOTS_26
    v.reset([seed + i for i in range(v.count)], DECK_26, DECK_26, slots, slots, setup, setup)
    my_seat = np.arange(v.count) % 2
    idx = np.arange(v.count)
    res = {"games": 0, "wins": 0, "losses": 0, "draws": 0, "cut": 0, "lengths": []}
    ep = np.zeros(v.count, dtype=int)
    if replays:
        Path(replays).mkdir(parents=True, exist_ok=True)
    while res["games"] < games:
        obs, masks = v.observations, v.masks
        v.actions[idx, my_seat] = me(obs[idx, my_seat], masks[idx, my_seat])
        v.actions[idx, 1 - my_seat] = opp(obs[idx, 1 - my_seat], masks[idx, 1 - my_seat])
        v.step()
        ep += 1
        info = v.infos
        for i in np.flatnonzero((v.dones != 0) & (info[:, s.info["reset"]] == 0)):
            if res["games"] >= games:
                break
            res["games"] += 1
            res["lengths"].append(int(ep[i]))
            ep[i] = 0
            w = int(info[i, s.info["winner"]])
            if replays:
                name = f"game_{res['games']:04d}_seat{my_seat[i]}_winner{w}.json"
                (Path(replays) / name).write_text(v.replay_json(int(i)))
            if w < 0:
                res["cut"] += 1
            elif w == 2:
                res["draws"] += 1
            elif w == my_seat[i]:
                res["wins"] += 1
            else:
                res["losses"] += 1
    n = max(1, res["games"])
    res["win_rate"] = res["wins"] / n
    res["score"] = (res["wins"] + 0.5 * res["draws"]) / n
    res["mean_length"] = float(np.mean(res["lengths"])) if res["lengths"] else 0.0
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--opponent", required=True, help=".pt path, scripted or random")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--install", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--level", type=int, default=16)
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--save-replays", default=None, help="folder where the game replays are written")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    args.tables = str(Path(args.tables).expanduser())
    dev = args.device

    model, _ = load(args.model, dev)
    model.eval()
    config = LadderConfig(arena="Arena_competitive" if args.save_replays else None)
    v = VectorBattle(args.envs, args.threads, install=args.install, tables=args.tables, config=config)
    ci = v.schema.header["entity_count"][0]
    me = model_policy(model, dev, ci, args.temperature)
    opp = make_opponent(args.opponent, v, dev, ci, args.temperature, args.seed)
    r = run_games(v, me, opp, args.games, args.seed, args.level, args.plain, args.save_replays)
    print(f"{r['games']} games vs {args.opponent}: {r['wins']} wins ({r['win_rate']:.1%}), "
          f"{r['losses']} losses, {r['draws']} draws, {r['cut']} cut | mean length "
          f"{r['mean_length']:.0f} decisions", flush=True)
    v.close()


if __name__ == "__main__":
    main()
