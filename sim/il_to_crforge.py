"""IL_Replay to crforge imitation data (2.6 games already filtered, or all games with --all-sides).

Each game is replayed in crforge: the plays of both players are given as actions, one
decision at a time (5 ticks), and for each 2.6 side we keep the observation, the mask and the
play. Several games are replayed at once in a VectorBattle.

What IL_Replay does not give, and what we do instead:
- the starting hand and the cycle order: deduced from the order of the cards played;
- the seed: any seed works, we reorder the deck so that the crforge shuffle deals exactly
  the deduced hand;
- the rest of the battle randomness (directions, delays) is not the one of the real game, so
  the state drifts a little as the game goes on.
A play that crforge refuses (elixir, occupied tile) is retried for --max-delay ticks, then
dropped. Each row keeps the number of plays dropped so far, so they can be filtered later.

Usage (from the project root):
  python sim/il_to_crforge.py --replays data/replays/il_replay_26/hog26_replays.parquet \
      --out data/crforge_bc --install sim/crforge/env/build/install/env \
      --tables ~/.crforge/tables/16.635.8 --threads 20 --batch 256
To split the work over several processes: --shard 0 --num-shards 4, then 1, 2, 3.

All games, both players (pretraining on all decks):
  python sim/il_to_crforge.py --replays data/replays/il_replay_all/part-*.parquet --all-sides \
      --out data/il_all --keep-wait 0.05 ... --shard 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cards import TOWERS, card_name, slot_flag  # noqa: E402

from crforge_env import LadderConfig, SideSetup, VectorBattle  # noqa: E402

SIDES = ("team", "opponent")  # team = side 0 (bottom), opponent = side 1 (top)


@dataclass
class Play:
    tick: int  # tick of the play, in 1/20 s
    card: str | None  # None for a hero ability
    x: int = 0
    y: int = 0


@dataclass
class Side:
    names: list[str]
    slots: list[int]
    levels: list[int]
    tower_key: str
    tower_level: int
    plays: list[Play] = field(default_factory=list)
    abilities: list[Play] = field(default_factory=list)
    order: list[str] = field(default_factory=list)  # deduced deal order
    order_fits: int = 0  # number of plays that fit this order


@dataclass
class Match:
    index: int
    tag: str
    sides: tuple[Side, Side]
    seats26: list[int]
    result: str
    crowns: tuple[int, int]
    last_tick: int


def deal_order(deck: list[str], plays: list[str]) -> tuple[list[str], int]:
    """The deal order (4 in hand, then the queue) that explains the most plays.

    A played card goes to the back of the queue, and the first card of the queue enters the hand.
    """
    best, best_fit = list(deck), -1
    for hand in combinations(deck, 4):
        if plays and plays[0] not in hand:
            continue
        rest = [c for c in deck if c not in hand]
        for queue in permutations(rest):
            held, waiting, fit = set(hand), list(queue), 0
            for card in plays:
                if card not in held:
                    break
                held.remove(card)
                waiting.append(card)
                held.add(waiting.pop(0))
                fit += 1
            if fit > best_fit:
                best, best_fit = [*hand, *queue], fit
                if fit == len(plays):
                    return best, fit
    return best, best_fit


def parse_match(index: int, row: dict, tick_offset: int, all_sides: bool = False) -> Match | str:
    """A row of the filtered parquet turned into a game to replay, or the reason it is refused."""
    payload = json.loads(row["payload_json"])
    battle = payload["battle"]
    sides: list[Side] = []
    for side in SIDES:
        players = battle[side]["players"]
        if len(players) != 1:
            return "not one player per side"
        deck = players[0]["deck"]
        names = [card_name(c["card_key"]) for c in deck]
        if len(names) != 8 or None in names or len(set(names)) != 8:
            return "unknown or incomplete deck"
        tower = players[0].get("tower_card") or {}
        sides.append(
            Side(
                names=names,
                slots=[slot_flag(c["card_key"]) for c in deck],
                levels=[int(c["level"]) for c in deck],
                tower_key=tower.get("card_key", "tower-princess"),
                tower_level=int(tower.get("level") or 0),
            )
        )
    last_tick = 0
    events = sorted(payload["events"], key=lambda e: (e["replay_tick_20hz"], e.get("source_index", 0)))
    for e in events:
        s = sides[SIDES.index(e["side"])]
        tick = int(e["replay_tick_20hz"]) + tick_offset
        last_tick = max(last_tick, tick)
        if e["kind"] == "activate_ability":
            s.abilities.append(Play(tick, None))
            continue
        name = card_name(e["card_key"])
        if name not in s.names:
            return "played card not in the deck"
        p = e["coordinates"]["native_world_units"]
        s.plays.append(Play(tick, name, int(p["x"]), int(p["y"])))
    for s in sides:
        s.order, s.order_fits = deal_order(s.names, [p.card for p in s.plays])
    if all_sides:  # both players, whatever their deck
        seats = [0, 1]
    else:
        seats = [SIDES.index(x) for x in row["hog26_sides"].split(",") if x]
    return Match(
        index=index,
        tag=row["replay_tag"],
        sides=(sides[0], sides[1]),
        seats26=seats,
        result=row["result"],
        crowns=(int(row["team_crowns"]), int(row["opponent_crowns"])),
        last_tick=last_tick,
    )


def tile_candidates(x: int, y: int, seat: int) -> list[tuple[int, int]]:
    """The tiles (col, row) in the seat's frame to try for a point on the arena.

    Seat 0 frame: x' = 18000 - x, y' = y. Seat 1: x' = x, y' = 32000 - y.
    A play is placed at the center of a tile; a 2x2 building (Cannon) on a corner.
    """
    px, py = (18000 - x, y) if seat == 0 else (x, 32000 - y)
    fx, fy = px / 1000.0, py / 1000.0
    if abs(fx - round(fx)) < 0.1 and abs(fy - round(fy)) < 0.1:
        cx, cy = round(fx), round(fy)
        cand = [(cx, cy), (cx - 1, cy), (cx, cy - 1), (cx - 1, cy - 1)]
    else:
        cand = [(int(fx), int(fy))]
    return [(c, r) for c, r in cand if 0 <= c < 18 and 0 <= r < 32]


def nearest_open(mask_slot: np.ndarray, x: int, y: int, seat: int, radius: int = 2) -> int | None:
    """The open tile closest to the point (at most `radius` tiles away), as an index 0..575."""
    px, py = (18000 - x, y) if seat == 0 else (x, 32000 - y)
    grid = mask_slot.reshape(32, 18)
    rows, cols = np.nonzero(grid)
    if len(rows) == 0:
        return None
    d = np.hypot(cols * 1000 + 500 - px, rows * 1000 + 500 - py)
    k = int(np.argmin(d))
    if d[k] > radius * 1000 + 500:
        return None
    return int(rows[k] * 18 + cols[k])


class Replayer:
    """Replays batches of games in a VectorBattle and keeps the imitation rows."""

    def __init__(self, args):
        self.args = args
        config = LadderConfig(decision_ticks=5)
        self.v = VectorBattle(args.batch, args.threads, install=args.install, tables=args.tables, config=config)
        self.check = VectorBattle(1, 1, install=args.install, tables=args.tables, config=config)
        # self.v keeps autoreset on: a finished game restarts by itself and is ignored after that
        self.check.set_autoreset(False)
        s = self.v.schema
        self.s = s
        self.ids = {n: i for i, n in enumerate(self.v.vocabulary)}
        self.h_hand = list(s.header["hand_card"])
        self.h_queue = list(s.header["queue"])
        self.first_ability = s.first_ability
        self.rng = np.random.default_rng(args.seed)

    def setups(self, m: Match) -> list[SideSetup | None]:
        """The towers and levels of each side, or None if crforge refuses the ones from the game."""
        out = []
        for side_index, side in enumerate(m.sides):
            setup = SideSetup(
                towers=TOWERS.get(side.tower_key, "King_PrincessTowers"),
                king_level=side.tower_level or None,
                tower_level=side.tower_level or None,
                card_levels=side.levels,
            )
            try:
                both = [None, None]
                both[side_index] = setup
                self.check.reset([1], m.sides[0].names, m.sides[1].names, None, None, both[0], both[1])
                out.append(setup)
            except ValueError:
                out.append(None)
        return out

    def _dealt(self, i: int, seat: int) -> list[str]:
        o = self.v.observations[i, seat]
        voc = self.v.vocabulary
        return [voc[k] if k >= 0 else "" for k in [*o[self.h_hand], *o[self.h_queue]]]

    def _reset(self, decks, slots, setups, seeds):
        n = self.args.batch
        pad = n - len(decks[0])
        d0 = decks[0] + [decks[0][0]] * pad
        d1 = decks[1] + [decks[1][0]] * pad
        s0 = slots[0] + [slots[0][0]] * pad
        s1 = slots[1] + [slots[1][0]] * pad
        u0 = setups[0] + [None] * pad
        u1 = setups[1] + [None] * pad
        self.v.reset(seeds + [1] * pad, d0, d1, s0, s1, u0, u1)

    def arrange(self, batch: list[Match], setups):
        """Reorders each deck so that crforge deals the deduced order. Returns decks, slots, setups."""
        decks = [[list(m.sides[k].names) for m in batch] for k in (0, 1)]
        flags = [[dict(zip(m.sides[k].names, m.sides[k].slots)) for m in batch] for k in (0, 1)]
        levels = [[dict(zip(m.sides[k].names, m.sides[k].levels)) for m in batch] for k in (0, 1)]
        seeds = [1 + m.index for m in batch]
        ok = [[False, False] for _ in batch]

        def build():
            sl = [[[flags[k][j][c] for c in decks[k][j]] for j in range(len(batch))] for k in (0, 1)]
            su = [[], []]
            for k in (0, 1):
                for j in range(len(batch)):
                    base = setups[j][k]
                    if base is None:
                        su[k].append(None)
                    else:
                        su[k].append(SideSetup(base.towers, base.king_level, base.tower_level,
                                               [levels[k][j][c] for c in decks[k][j]]))
            return sl, su

        for _ in range(4):
            sl, su = build()
            self._reset(decks, sl, su, seeds)
            changed = False
            for j, m in enumerate(batch):
                for k in (0, 1):
                    want = m.sides[k].order
                    got = self._dealt(j, k)
                    if set(got[:4]) == set(want[:4]) and got[4:] == want[4:]:
                        ok[j][k] = True
                        continue
                    ok[j][k] = False
                    # got[i] = deck[pi[i]]  ->  put want[i] at index pi[i]
                    pi = [decks[k][j].index(c) for c in got]
                    new = [""] * 8
                    for pos, card in zip(pi, want):
                        new[pos] = card
                    decks[k][j] = new
                    changed = True
            if not changed:
                break
        sl, su = build()
        return decks, sl, su, seeds, ok

    def run(self, batch: list[Match]):
        a = self.args
        s, v = self.s, self.v
        setups = [self.setups(m) for m in batch]
        decks, slots, sups, seeds, deal_ok = self.arrange(batch, setups)
        self._reset(decks, slots, sups, seeds)

        n = len(batch)
        plays = [[deque(m.sides[k].plays) for k in (0, 1)] for m in batch]
        abil = [[deque(m.sides[k].abilities) for k in (0, 1)] for m in batch]
        stats = [{"given": [0, 0], "late": [0, 0], "dropped": [0, 0], "abilities": [0, 0],
                  "abilities_dropped": [0, 0], "rows": 0} for _ in batch]
        active = np.zeros(v.count, dtype=bool)
        active[:n] = True
        end_tick = [m.last_tick + 200 for m in batch]
        rows = {"obs": [], "mask": [], "action": [], "tick": [], "match": [], "seat": [],
                "delay": [], "dropped": []}
        tick = 0
        while active.any():
            obs, masks = v.observations, v.masks
            v.actions[:] = 0
            for j in np.flatnonzero(active[:n]):
                for seat in (0, 1):
                    act, delay = self._choose(j, seat, tick, obs, masks, plays, abil, stats)
                    v.actions[j, seat] = act
                    if seat in batch[j].seats26 and (act != 0 or self.rng.random() < a.keep_wait):
                        rows["obs"].append(obs[j, seat].copy())
                        rows["mask"].append(np.packbits(masks[j, seat]))
                        rows["action"].append(act)
                        rows["tick"].append(tick)
                        rows["match"].append(batch[j].index)
                        rows["seat"].append(seat)
                        rows["delay"].append(delay)
                        rows["dropped"].append(sum(stats[j]["dropped"]))
                        stats[j]["rows"] += 1
            v.step()
            tick += 5
            for j in np.flatnonzero(active[:n]):
                info = v.infos[j]
                if v.dones[j] or info[5] != 0 or tick > end_tick[j]:
                    active[j] = False
                    stats[j]["sim_winner"] = int(info[2])
                    stats[j]["sim_crowns"] = [int(info[3]), int(info[4])]
                    stats[j]["fault"] = int(info[5])
                    stats[j]["end_tick"] = tick
        for j, m in enumerate(batch):
            left = [len(plays[j][k]) for k in (0, 1)]
            stats[j]["dropped"] = [stats[j]["dropped"][k] + left[k] for k in (0, 1)]
            stats[j].update(tag=m.tag, index=m.index, seats26=m.seats26, deal_ok=deal_ok[j],
                            order_fits=[m.sides[k].order_fits for k in (0, 1)],
                            plays=[len(m.sides[k].plays) for k in (0, 1)],
                            setups=[x is not None for x in setups[j]],
                            result=m.result, crowns=list(m.crowns))
        return rows, stats

    def _choose(self, j, seat, tick, obs, masks, plays, abil, stats):
        """The seat's action at this decision, and the delay of the play (-1 for a wait)."""
        a = self.args
        st = stats[j]
        queue = plays[j][seat]
        while queue and tick - queue[0].tick > a.max_delay:
            queue.popleft()
            st["dropped"][seat] += 1
        mask = masks[j, seat]
        if queue and queue[0].tick < tick + 5:
            p = queue[0]
            hand = obs[j, seat, self.h_hand]
            where = np.flatnonzero(hand == self.ids[p.card])
            if len(where):
                slot = int(where[0])
                base = 1 + slot * 576
                block = mask[base: base + 576]
                if block.any():
                    choice = None
                    for c, r in tile_candidates(p.x, p.y, seat):
                        if block[r * 18 + c]:
                            choice = r * 18 + c
                            break
                    if choice is None:
                        choice = nearest_open(block, p.x, p.y, seat)
                    if choice is not None:
                        queue.popleft()
                        delay = max(0, tick - p.tick)
                        st["given"][seat] += 1
                        st["late"][seat] += int(delay > 0)
                        return base + choice, delay
        aq = abil[j][seat]
        while aq and tick - aq[0].tick > a.max_delay:
            aq.popleft()
            st["abilities_dropped"][seat] += 1
        if aq and aq[0].tick < tick + 5:
            for k in (0, 1):
                act = self.first_ability + k
                if mask[act]:
                    p = aq.popleft()
                    st["abilities"][seat] += 1
                    return act, max(0, tick - p.tick)
        return 0, -1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", nargs="+", required=True,
                    help="parquet of the 2.6 games (scripts/filter_26.py), or the raw IL_Replay files with --all-sides")
    ap.add_argument("--all-sides", action="store_true", help="keep both players of each game, all decks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--install", required=True, help="sim/crforge/env/build/install/env")
    ap.add_argument("--tables", required=True, help="~/.crforge/tables/16.635.8")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64, help="games replayed at once")
    ap.add_argument("--limit", type=int, default=0, help="max number of games (0 = all)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--tick-offset", type=int, default=0, help="offset added to the IL_Replay ticks")
    ap.add_argument("--max-delay", type=int, default=20, help="ticks during which a refused play is retried")
    ap.add_argument("--keep-wait", type=float, default=0.1, help="share of the decisions without a play that are kept")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0, help="stop cleanly between two batches after this time")
    args = ap.parse_args()
    t_begin = time.time()  # --max-hours counts from the start (reading the files included)
    args.tables = str(Path(args.tables).expanduser())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    matches, refused = [], {}
    index = -1
    for path in sorted(args.replays):  # global numbering, in file order
        if args.limit and len(matches) >= args.limit:
            break
        for b in pq.ParquetFile(path).iter_batches(batch_size=1000):
            for row in b.to_pylist():
                index += 1
                if index % args.num_shards != args.shard:
                    continue
                try:
                    m = parse_match(index, row, args.tick_offset, args.all_sides)
                except (KeyError, TypeError, ValueError) as e:  # malformed row in the raw files
                    m = f"unreadable row ({type(e).__name__})"
                if isinstance(m, str):
                    refused[m] = refused.get(m, 0) + 1
                else:
                    matches.append(m)
                if args.limit and len(matches) >= args.limit:
                    break
            if args.limit and len(matches) >= args.limit:
                break
    print(f"{len(matches)} games to replay, refused: {refused}", flush=True)

    rep = Replayer(args)
    rep.v.write_metadata(out / "metadata.json")
    (out / "convert_args.json").write_text(json.dumps(vars(args), indent=1))
    t0 = time.time()
    total = {"given": 0, "late": 0, "dropped": 0, "rows": 0, "plays": 0}
    tag = f"s{args.shard:02d}"
    deadline = t_begin + args.max_hours * 3600 if args.max_hours else None
    with open(out / f"matches_{tag}.jsonl", "w") as log:
        for start in range(0, len(matches), args.batch):
            if deadline and time.time() > deadline:
                print(f"time limit reached: clean stop after {start} games", flush=True)
                break
            batch = matches[start: start + args.batch]
            rows, stats = rep.run(batch)
            if rows["obs"]:
                final = out / f"rows_{tag}_{start:06d}.npz"
                tmp = out / f"rows_{tag}_{start:06d}.tmp"  # written then renamed: never a half-written file
                with open(tmp, "wb") as fh:
                    np.savez_compressed(
                        fh,
                        obs=np.stack(rows["obs"]).astype(np.int32),
                        mask=np.stack(rows["mask"]).astype(np.uint8),
                        action=np.array(rows["action"], dtype=np.int32),
                        tick=np.array(rows["tick"], dtype=np.int32),
                        match=np.array(rows["match"], dtype=np.int32),
                        seat=np.array(rows["seat"], dtype=np.int8),
                        delay=np.array(rows["delay"], dtype=np.int16),
                        dropped=np.array(rows["dropped"], dtype=np.int16),
                    )
                tmp.replace(final)
            for st in stats:
                log.write(json.dumps(st) + "\n")
                for seat in st["seats26"]:
                    total["given"] += st["given"][seat]
                    total["late"] += st["late"][seat]
                    total["dropped"] += st["dropped"][seat]
                    total["plays"] += st["plays"][seat]
                total["rows"] += st["rows"]
            log.flush()  # the games of this batch are logged before the next one starts
            done = start + len(batch)
            rate = done / (time.time() - t0)
            p = max(1, total["plays"])
            print(f"{done}/{len(matches)} games, {rate:.1f}/s | {'kept players' if args.all_sides else '2.6 side'}: plays given "
                  f"{total['given'] / p:.1%} (late {total['late'] / p:.1%}), dropped "
                  f"{total['dropped'] / p:.1%} | rows {total['rows']}", flush=True)
    rep.v.close()
    rep.check.close()


if __name__ == "__main__":
    main()
