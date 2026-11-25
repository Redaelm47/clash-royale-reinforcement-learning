"""The twin: the real match replayed live in crforge, to give the bot its FULL observation.

The screen does not show everything (troop HP, targets, projectiles, freezes, timers...). crforge
computes all of it. We give it the plays of the real match, yours (known exactly) and the
opponent's (spotted by the vision: card, point, time), and it replays the match up to now. The
bot then gets the crforge observation, the same as in training, where we replace what the screen
knows better (your hand, your elixir, the tower HP and state).

How crforge accepts our plays: it only plays a card that is in the hand it dealt. So we order each
deck so that, from the start, it deals an order that explains every play seen (the played card
goes to the back of the queue), and we shift a play by one or two ticks when the twin does not
have enough elixir or hand refill for it yet.

Measured on the Mac: a rebuild takes 25 to 60 ms (needed after each new play), and stepping one
decision forward takes 0.1 ms.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from state import BASE_MASK, DECK_NAMES, EVO_NEED, INFO, SLOT_FLAGS  # noqa: E402

TABLES = HERE / "model" / "tables"
INSTALL = HERE / "crforge_lite" / "env"
TEMPLATE = HERE / "model" / "twin_template.json"
IDS = INFO["ids"]
RARITY = INFO["rarity"]
COST = {n: c["cost"] for n, c in INFO["cards"].items()}
RARITY_START = {"Common": 1, "Rare": 3, "Epic": 6, "Legendary": 9, "Champion": 11}
SETTINGS_FILE = HERE / "settings.json"


def settings():
    s = {"my_card_level": 11, "my_tower_level": 11, "enemy_card_level": 11, "enemy_tower_level": 11,
         "my_evolutions": ["Skeletons"], "my_heroes": [], "enemy_evolutions": [], "enemy_heroes": []}
    try:
        s.update(json.loads(SETTINGS_FILE.read_text()))
    except Exception:
        pass
    return s


def level_index(card, level):
    """A card level the way a replay writes it: counted from the first level of its rarity."""
    return max(0, level - RARITY_START.get(RARITY.get(card), 3))
REGEN = 10000 / 56          # elixir (in ten-thousandths) per tick, before double elixir
START_OPEN = 90             # no card before tick 90 (crforge start delay)


def deal_order(deck, plays):
    """The deal order (4 in hand, then the queue) that explains the most plays."""
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


def explain(deck, events, max_insert=2):
    """Like deal_order, but for plays seen by the vision, so incomplete:
    - None = a play whose card we don't know (a deploy clock with no recognized troop);
    - a card that cannot be in hand (a play was missed before): we insert up to `max_insert`
      hidden plays to bring it back, instead of losing all the following plays.
    Returns (start order, plan) where plan = [(event index or None if inserted, card)]."""
    best = (None, None, -1e9)
    seq = list(events)
    for hand in combinations(deck, 4):
        if seq and seq[0] is not None and seq[0] not in hand and max_insert == 0:
            continue
        rest = [c for c in deck if c not in hand]
        for queue in permutations(rest):
            h, q, plan, fit, ins = list(hand), list(queue), [], 0, 0

            def play(card):
                h.remove(card)
                q.append(card)
                h.append(q.pop(0))

            def spare(i, avoid=None):
                """The card in hand we need the least soon."""
                upcoming = [e for e in seq[i + 1:] if e is not None]
                cands = [c for c in h if c != avoid]
                return max(cands, key=lambda c: upcoming.index(c) if c in upcoming else 99)
            for i, c in enumerate(seq):
                if c is None:
                    pick = spare(i)
                    play(pick)
                    plan.append((i, pick))
                elif c in h:
                    play(c)
                    plan.append((i, c))
                    fit += 1
                elif c in q and q.index(c) < max_insert:
                    for _ in range(q.index(c) + 1):
                        pick = spare(i, avoid=c)
                        play(pick)
                        plan.append((None, pick))
                        ins += 1
                    play(c)
                    plan.append((i, c))
                    fit += 1
                else:
                    plan.append((i, None))
            score = fit - 0.6 * ins
            if score > best[2]:
                best = ([*hand, *queue], plan, score)
                if ins == 0 and fit == sum(1 for e in seq if e is not None):
                    return best[0], best[1]
    return best[0], best[1]


def refill_ticks(tick):
    return 20 if tick < 2400 else (10 if tick < 4800 else 7)


def regen(tick):
    return REGEN * (2 if tick >= 2400 else 1)


def schedule(order, plays):
    """Ticks of each play of one side that crforge accepts: card in hand, enough elixir and
    refill. plays: list of (card, target tick). Returns [(play index, tick)]."""
    hand = list(order[:4])
    queue = list(order[4:])
    pending = [None] * 4           # slot -> run tick of the pending play
    empty_until = [0] * 4
    runs = []                      # (run tick, slot, card, cost)
    elixir, reserved, t_now = 60000.0, 0.0, 0
    last_refill, last_given = -1000, -1000
    out = []

    def advance(t):
        nonlocal elixir, reserved, t_now, last_refill
        while True:
            due = [r for r in runs if r[0] <= t]
            nxt = min(due, key=lambda r: r[0]) if due else None
            stop = nxt[0] if nxt else t
            while t_now < stop:  # regen tick by tick (double elixir changes the rate)
                elixir = min(100000.0, elixir + regen(t_now))
                t_now += 1
            if nxt is None:
                return
            runs.remove(nxt)
            run, slot, card, cost = nxt
            elixir = max(0.0, elixir - cost * 10000)
            reserved -= cost * 10000
            pending[slot] = None
            queue.append(card)
            new = queue.pop(0)
            cd = refill_ticks(run)
            if run - last_refill >= cd:
                last_refill = run
                hand[slot] = new
            else:
                last_refill += cd
                empty_until[slot] = last_refill
                hand[slot] = new

    for i, (card, want) in enumerate(plays):
        t = max(int(round(want / 5.0)) * 5, START_OPEN, last_given + 5)
        for _ in range(80):  # up to 400 ticks late, otherwise we drop this play
            advance(t)
            slot = next((k for k in range(4) if hand[k] == card), None)
            cost = COST.get(card, 3)
            if (slot is not None and pending[slot] is None and t >= empty_until[slot]
                    and elixir - reserved >= cost * 10000):
                pending[slot] = t + 20
                reserved += cost * 10000
                runs.append((t + 20, slot, card, cost))
                out.append((i, t))
                last_given = t
                break
            t += 5
    return out


class Twin:
    """Several crforge matches in parallel ("variants"), rebuilt after each new play.

    Variant 0 takes the enemy plays as the vision saw them. The others shift them a little (time,
    point) and draw another game randomness. On each frame every variant is scored on what the
    screen shows (troops seen or not, tower HP), and the bot gets the best one."""

    def __init__(self, n=4, background=True):
        sys.path.insert(0, str(HERE))
        import os
        import threading
        from crforge_lite import LadderConfig, VectorBattle
        self.n = n
        threads = max(1, min(n, (os.cpu_count() or 4) - 2))
        self.v = VectorBattle(n, threads, install=str(INSTALL), tables=str(TABLES),
                              config=LadderConfig(arena="Arena_competitive"))
        # background rebuilds happen in a second vector, swapped in when ready, so the bot never
        # waits for them (before this: blocked up to 1.5 s on a laptop)
        self.background = background
        self.v_build = VectorBattle(n, threads, install=str(INSTALL), tables=str(TABLES),
                                    config=LadderConfig(arena="Arena_competitive")) if background else None
        self.job = None
        self.job_lock = threading.Lock()
        self.build_ms = []
        self.retry_after = 0.0
        meta = json.loads((HERE / "model" / "metadata.json").read_text())
        self.names = meta["vocabulary"]["names"]
        self.H = {}
        for i, f in enumerate(meta["schema"]["header_fields"]):
            self.H.setdefault(f, []).append(i)
        self.E = {f: i for i, f in enumerate(meta["schema"]["entity_fields"])}
        self.template = json.loads(TEMPLATE.read_text())
        self.seeds = [int(self.template["rndSeed"]) + 7919 * i for i in range(n)]
        self.own = []          # (card, time given (PC clock), x, y) in tiles, in the bot's frame
        self.enemy = []        # (card, time given, x, y, form seen)
        self.t0 = None         # match start (PC clock), corrected from the elixir
        self.cfg = settings()
        self.own_flags = [(2 if c in self.cfg["my_heroes"] else (1 if c in self.cfg["my_evolutions"] else 0))
                          for c in DECK_NAMES]
        self.enemy_deck = list(DECK_NAMES)
        self.enemy_flags = [(2 if c in self.cfg["enemy_heroes"] else (1 if c in self.cfg["enemy_evolutions"] else 0))
                            for c in DECK_NAMES]
        self.dirty = True
        self.tick = None
        self.next_due = float("inf")  # tick of the next known play that is not in the twin yet
        self.ok = False
        self.alive = [True] * n       # usable variants
        self.score = [0.0] * n        # score of each variant (moving average)
        self.best = 0
        self.dropped = [0, 0]
        self.rebuilds = 0
        self.errors = []
        self.pi = {}
        self.probed = None
        self.last_seen = {}
        self.inferred = 0
        self._probe(self.v)
        # HP of the same unit at level 16 and at the real level (the same factor for all, ~1.6 from 11 to 16)
        f16 = self.v.unit_facts(16)
        self.k = {}
        for side, lvl in ((0, self.cfg["my_card_level"]), (1, self.cfg["enemy_card_level"])):
            fl = self.v.unit_facts(lvl)
            r = [f16[c].hitpoints / fl[c].hitpoints for c in f16 if c in fl and fl[c].hitpoints > 50]
            self.k[side] = float(np.median(r)) if r else 1.0
        self.hp16 = {f.unit: f.hitpoints for f in f16.values()}

    # ---- the crforge deal for each seed ----
    def _probe(self, vec, enemy_deck=None, enemy_flags=None):
        """crforge always deals the same deck positions for a given seed: we read them (each time
        the deck changes), then order each deck to get the order we want."""
        enemy_deck = enemy_deck or self.enemy_deck
        enemy_flags = enemy_flags or self.enemy_flags
        key = (tuple(enemy_deck), tuple(enemy_flags), tuple(self.own_flags))
        if self.probed == key:
            return
        pi = {}
        f0 = dict(zip(DECK_NAMES, self.own_flags))
        f1 = dict(zip(enemy_deck, enemy_flags))
        reps = [self._replay([], DECK_NAMES, f0, enemy_deck, f1, end=100, env=i, pi={}) for i in range(self.n)]
        vec.start_from_replays(list(range(self.n)), reps, [5] * self.n)
        for i in range(self.n):
            pi[i] = {}
            for seat, deck in ((0, DECK_NAMES), (1, enemy_deck)):
                o = vec.observations[i, seat]
                got = [self.names[k] for k in [*o[self.H["hand_card"]], *o[self.H["queue"]]]]
                pi[i][seat] = [deck.index(c) for c in got]
        self.pi, self.probed = pi, key

    def _entry(self, card, flags_of, level):
        e = {"d": IDS[card], "l": level_index(card, level)}
        if flags_of.get(card):
            e["el"] = flags_of[card]
        return e

    def _replay(self, cmds, order0, f0, order1, f1, end, env=0, pi=None):
        pi = self.pi if pi is None else pi
        r = copy.deepcopy(self.template)
        r["rndSeed"] = self.seeds[env]
        lv = {0: self.cfg["my_card_level"], 1: self.cfg["enemy_card_level"]}
        for seat, order, fl in ((0, order0, f0), (1, order1, f1)):
            key = f"deck{seat}"
            if env in pi:
                sp = [None] * 8
                for pos, card in zip(pi[env][seat], order):
                    sp[pos] = self._entry(card, fl, lv[seat])
            else:  # first probe: the order as is
                sp = [self._entry(c, fl, lv[seat]) for c in order]
            r["battle"][key]["sp"] = sp
            tower = self.cfg["my_tower_level"] if seat == 0 else self.cfg["enemy_tower_level"]
            for sc in r["battle"][key].get("sc", []):
                sc["l"] = tower - 1
            r["battle"]["hbd"][seat]["kt"] = tower
        r["cmd"] = cmds
        r["endTick"] = end
        return r

    def reset(self):
        self.own, self.enemy = [], []
        self.enemy_deck = list(DECK_NAMES)
        self.enemy_flags = [(2 if c in self.cfg["enemy_heroes"] else (1 if c in self.cfg["enemy_evolutions"] else 0))
                            for c in DECK_NAMES]
        self.t0, self.tick, self.ok, self.dirty = None, None, False, True
        self.next_due = float("inf")
        self.score = [0.0] * self.n
        self.best = 0

    # ---- the plays ----
    def add_own(self, card, now, tile):
        """Your play, given at time now (PC clock) on tile (col, row)."""
        self.own.append((card, now, tile[0] + 0.5, tile[1] + 0.5))
        self.dirty = True

    def remove_own(self, card):
        """Your last play of this card did not go through (refused by the game)."""
        for i in range(len(self.own) - 1, -1, -1):
            if self.own[i][0] == card:
                del self.own[i]
                self.dirty = True
                return

    def add_enemy(self, card, t_run, x, y, form=0):
        """An enemy play: the card (None if unknown: a clock with no recognized troop), the time its
        unit appeared (PC clock), its point."""
        if card is None:
            self.enemy.append((None, t_run - 1.0, x, y, 0))
            self.dirty = True
            return
        if card not in IDS:
            return
        if card not in self.enemy_deck:
            seen = {c for c, *_ in self.enemy}
            free = [i for i, c in enumerate(self.enemy_deck) if c not in seen]
            if not free:
                return
            i = free[-1]
            self.enemy_deck[i] = card
            self.enemy_flags[i] = 0
        if form == 1 and card in self.enemy_deck:  # an evolved play was seen: the card is an evolution for them
            self.enemy_flags[self.enemy_deck.index(card)] = 1
        self.enemy.append((card, t_run - 1.0, x, y, form))  # given 1 s (20 ticks) before it appears
        self.dirty = True

    def resolve(self, card, t_run, x, y):
        """The card of an unknown play (clock with no troop) is finally recognized."""
        for i, (c, t, ex, ey, form) in enumerate(self.enemy):
            if c is None and abs(ex - x) < 0.01 and abs(ey - y) < 0.01 and abs(t - (t_run - 1.0)) < 0.01:
                if card in IDS:
                    self.enemy[i] = (card, t, ex, ey, form)
                    if card not in self.enemy_deck:  # same rule as add_enemy
                        seen = {e[0] for e in self.enemy if e[0]}
                        free = [k for k, d in enumerate(self.enemy_deck) if d not in seen]
                        if free:
                            self.enemy_deck[free[-1]] = card
                            self.enemy_flags[free[-1]] = 0
                    self.dirty = True
                return

    def _cmds(self, side, plays, order, flags_of, deck_index, level):
        """Plays of one side in replay format, with ticks crforge accepts."""
        out = []
        evo = {}
        for i, t in schedule(order, [(p[0], p[1]) for p in plays]):
            card, _, x, y = plays[i][:4]
            fl = flags_of.get(card, 0)
            count, form = 0, 0
            if fl & 2:
                form = 2  # hero
            elif fl & 1:  # evolution: counter 1, 2, then 3 = evolved play
                n = evo.get(card, 0) + 1
                count = n
                form = 1 if n > EVO_NEED.get(card, 2) else 0
                evo[card] = 0 if form == 1 else n
            cost = COST.get(card, 3)
            pd = (cost << 28) | ((deck_index[card] + 1) << 22) | (fl << 19) | ((level - 1) << 10) | (count << 7) | form
            out.append({"ct": 153, "c": {"t": t, "t2": t + 20, "idHi": side, "idLo": 0,
                                         "px": int(round(18000 - x * 1000)), "py": int(round(y * 1000)),
                                         "sid": -1, "sel": {"os": IDS[card], "pd": pd}}})
        return out

    @staticmethod
    def _enemy_point(card, x, y):
        """An enemy drop point that crforge accepts: the closest allowed tile, from the crforge masks
        flipped (their side is ours seen from the other end of the arena)."""
        ctype = INFO["cards"].get(card, {}).get("type", "troop")
        if card == "Fireball" or (ctype == "spell" and card != "Log"):
            return min(17.5, max(0.5, int(x) + 0.5)), min(31.5, max(0.5, int(y) + 0.5))
        mask = BASE_MASK.get(card)
        if mask is None:
            mask = BASE_MASK["Cannon"] if ctype == "building" else BASE_MASK["HogRider"]
        col, row = min(17, max(0, int(x))), min(31, max(0, int(y)))
        ok = np.argwhere(mask[::-1, ::-1])  # (row, col) allowed for the opponent, in our frame
        d = (ok[:, 0] - row) ** 2 + (ok[:, 1] - col) ** 2
        r, c = ok[int(np.argmin(d))]
        return c + 0.5, r + 0.5

    @staticmethod
    def _jitter(i, env):
        """Fixed shift (ticks, tiles) of enemy play i in variant env (0 = as seen)."""
        if env == 0:
            return 0, 0.0, 0.0
        h = (i * 2654435761 + env * 40503) & 0xFFFFFFFF
        dt = (-6, -3, 3, 6)[h % 4]
        dx = (-1.0, 0.0, 0.0, 1.0)[(h >> 4) % 4]
        dy = (-1.0, 0.0, 0.0, 1.0)[(h >> 8) % 4]
        return dt, dx, dy

    # ---- the rebuild ----
    def sync(self, now, t0):
        """Brings the variants to time now of the real match started at t0. True if the best one is
        usable. After the first one, rebuilds run in the background: meanwhile the old one keeps
        stepping (0.1 ms per decision) and is swapped for the new one as soon as it is ready."""
        if t0 != self.t0:
            self.t0, self.dirty = t0, True
        target = max(START_OPEN + 10, int((now - t0) * 20) // 5 * 5)
        self._collect()
        need = (self.dirty or self.tick is None or target < self.tick or target - self.tick > 400
                or target > self.next_due)
        if need:
            import time as _time
            if self.tick is None or not self.background:
                self._apply(self._rebuild(self.v, target, self._snapshot()), self.v)
            elif self.job is None and _time.time() >= self.retry_after:
                self._launch(target)
        if self.ok and self.tick is not None:
            while self.tick < target:
                self.v.actions[:, :] = 0
                self.v.step()
                self.tick += 5
        return self.ok

    def _snapshot(self):
        self.dirty = False
        return {"own": list(self.own), "enemy": list(self.enemy), "enemy_deck": list(self.enemy_deck),
                "enemy_flags": list(self.enemy_flags), "t0": self.t0}

    def _launch(self, target):
        import threading
        snap = self._snapshot()
        job = {"done": threading.Event(), "res": None}

        def work():
            try:
                job["res"] = self._rebuild(self.v_build, target, snap)
            except Exception as e:  # never stop the bot for a failed rebuild
                job["res"] = {"ok": False, "target": target, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                self.dirty = True
            job["done"].set()
        self.job = job
        threading.Thread(target=work, daemon=True).start()

    def _collect(self):
        """A background rebuild is done: swap the two vectors."""
        job = self.job
        if job is None or not job["done"].is_set():
            return
        self.job = None
        res = job["res"]
        if res and res.get("ok"):
            self.v, self.v_build = self.v_build, self.v
            self._apply(res, self.v)
        else:  # keep the old twin (it keeps stepping) and retry in 2 s
            import time as _time
            self.retry_after = _time.time() + 2.0
            self.dirty = True
            if res and res.get("error"):
                self.errors.append(res["error"])

    def _apply(self, res, vec):
        self.next_due = res.get("next_due", float("inf"))
        if res.get("ok"):
            self.alive = res["alive"]
            self.tick = res["target"]
            self.ok = True
            self.rebuilds += 1
            self.dropped, self.inferred = res["dropped"], res["inferred"]
            self.build_ms = (self.build_ms + [res["ms"]])[-50:]
            if not self.alive[self.best]:
                self.best = next(i for i in range(self.n) if self.alive[i])
        else:
            self.ok = False
            self.tick = res.get("target", self.tick)

    def _rebuild(self, vec, target, snap):
        import time as _time
        t_start = _time.perf_counter()
        t0 = snap["t0"]
        tk = lambda t: (t - t0) * 20  # noqa: E731
        res = {"target": target, "ok": False}
        # a play only enters the twin once it is placed (20 ticks later): crforge refuses to start
        # while a play it had to force is still pending
        later = [tk(p[1]) + 21 for p in snap["own"] + snap["enemy"] if tk(p[1]) + 20 >= target]
        next_due = min(later) if later else float("inf")
        own = sorted([(c, tk(t), x, y) for c, t, x, y in snap["own"] if tk(t) < target], key=lambda p: p[1])
        enemy_deck, enemy_flags = snap["enemy_deck"], snap["enemy_flags"]
        self._probe(vec, enemy_deck, enemy_flags)
        pi = self.pi
        f0 = dict(zip(DECK_NAMES, self.own_flags))
        f1 = dict(zip(enemy_deck, enemy_flags))
        order0, _ = deal_order(DECK_NAMES, [p[0] for p in own])
        # the known enemy plays, in order, with unknown cards and missed plays inferred
        seen = sorted([(i, p) for i, p in enumerate(snap["enemy"]) if tk(p[1]) < target], key=lambda ip: ip[1][1])
        order1, plan = explain(enemy_deck, [p[0] for _, p in seen])
        planned = []  # (original index or None, card, time, x, y)
        for j, (ev, card) in enumerate(plan):
            if card is None:
                continue
            if ev is not None:
                i, (_, t, x, y, _) = seen[ev]
                planned.append((i, card, t, x, y))
            else:  # inferred hidden play: just before the play it explains, at a spot with no effect
                nxt = next((seen[e][1][1] for e, _ in plan[j:] if e is not None), None)
                t = (nxt - 0.6) if nxt is not None else t0
                spell = card in ("Fireball", "Arrows", "Zap", "Rocket", "Lightning", "Poison", "Snowball", "GiantSnowball")
                x, y = (0.5, 0.5) if spell else (0.5, 31.5)
                planned.append((None, card, t, x, y))
        res["inferred"] = sum(1 for e, c in plan if e is None) + sum(1 for _, p in seen if p[0] is None)
        res["dropped"] = [0, 0]

        def build(env, jitter=True):
            nonlocal next_due
            enemy = []
            for i, card, t, x, y in planned:
                dt, dx, dy = self._jitter(i, env) if (i is not None and jitter) else (0, 0.0, 0.0)
                if tk(t) + dt < target:
                    px, py = (x + dx, y + dy) if i is not None else (x, y)
                    enemy.append((card, tk(t) + dt, *self._enemy_point(card, px, py)))
            enemy.sort(key=lambda p: p[1])
            idx0 = {c: pi[env][0][i] for i, c in enumerate(order0)}
            idx1 = {c: pi[env][1][i] for i, c in enumerate(order1)}
            c0 = self._cmds(0, own, order0, f0, idx0, self.cfg["my_card_level"])
            c1 = self._cmds(1, enemy, order1, f1, idx1, self.cfg["enemy_card_level"])
            if env == 0 and jitter:
                res["dropped"] = [len(own) - len(c0), len(enemy) - len(c1)]
            cmds = sorted([c for c in c0 + c1 if c["c"]["t2"] < target], key=lambda c: (c["c"]["t"], c["c"]["idHi"]))
            late = [c["c"]["t2"] + 1 for c in c0 + c1 if c["c"]["t2"] >= target]
            if late:
                next_due = min(next_due, min(late))
            return self._replay(cmds, order0, f0, order1, f1, end=max(target + 100, 4000), env=env, pi=pi)
        reps = [build(env) for env in range(self.n)]
        plain = set()
        alive = list(range(self.n))
        for _ in range(10):
            try:
                vec.start_from_replays(alive, [reps[i] for i in alive], [target] * len(alive))
                break
            except ValueError as e:  # a play or a variant that crforge refuses: remove it and try again
                msg = str(e)
                self.errors.append(msg[:300])
                m_env = re.search(r"environment (\d+)", msg)
                m_cmd = re.search(r"cmd\[(\d+)\]", msg)
                env = int(m_env.group(1)) if m_env else None
                if env is not None and env not in plain and env != 0:
                    reps[env] = build(env, jitter=False)  # the variant without its shifts
                    plain.add(env)
                elif env is not None and m_cmd and reps[env]["cmd"]:
                    del reps[env]["cmd"][int(m_cmd.group(1))]
                elif env is not None and env in alive:
                    alive.remove(env)
                else:
                    alive = []
                if not alive:
                    break
        res["alive"] = [i in alive for i in range(self.n)]
        res["ok"] = bool(alive)
        res["next_due"] = next_due
        res["ms"] = round((_time.perf_counter() - t_start) * 1000, 1)
        return res

    # ---- which variant fits the screen best ----
    def judge(self, screen_units, tower_frac, now):
        """Scores each variant: troops seen on screen that it has (+1), reliable troops it thinks are
        alive but the screen does not see (-1), tower HP gap (-2 per full tower)."""
        if not self.ok:
            return
        H, E = self.H, self.E
        seen = [(u["side"], family(u["row"]), u["x"], u["y"]) for u in screen_units if u.get("seen", 3) >= 2]
        for env in range(self.n):
            if not self.alive[env]:
                self.score[env] = -1e9
                continue
            o = self.v.observations[env, 0]
            n = int(o[H["entity_count"][0]])
            units = []
            for k in range(6, n):
                a = 224 + k * 34
                if o[a + E["kind"]] != 1:
                    continue
                row = int(o[a + E["row"]])
                units.append((int(o[a + E["side"]]), family(self.names[row]) if 0 <= row < len(self.names) else "",
                              o[a + E["x"]] / 1000, o[a + E["y"]] / 1000, int(o[a + E["state"]])))
            hit = sum(1 for s, f, x, y in seen
                      if any(s == s2 and f == f2 and (x - x2) ** 2 + (y - y2) ** 2 < 6.25 for s2, f2, x2, y2, _ in units))
            ghost = sum(1 for s2, f2, x2, y2, st in units if f2 in RELIABLE and st != 4 and
                        not any(s == s2 and f == f2 and (x - x2) ** 2 + (y - y2) ** 2 < 6.25 for s, f, x, y in seen))
            tow = 0.0
            for k, frac in enumerate(tower_frac):
                a = 224 + k * 34
                mx = max(1, int(o[a + E["max_hp"]]))
                tow += abs(int(o[a + E["hp"]]) / mx - frac)
            sc = hit - ghost - 2.0 * tow
            self.score[env] = 0.85 * self.score[env] + 0.15 * sc
        live = [i for i in range(self.n) if self.alive[i]]
        cand = max(live, key=lambda i: self.score[i])
        if self.score[cand] > self.score[self.best] + 0.3 or not self.alive[self.best]:
            self.best = cand

    def observation(self):
        return self.v.observations[self.best, 0].copy()


# fields the screen knows better than the twin, or that must stay as in training
FROM_SCREEN = ["elixir", "elixir_reserved", "hand_refill_ms", "hand_card", "hand_cost", "hand_pending", "next_card",
               "queue", "pending_count", "pending_card", "pending_x", "pending_y", "pending_ticks", "deck_evolution",
               "hand_form", "crowns", "crowns_enemy", "own_king_hp", "own_left_hp", "own_right_hp", "enemy_king_hp",
               "enemy_left_hp", "enemy_right_hp", "own_king_max_hp", "own_left_max_hp", "own_right_max_hp",
               "enemy_king_max_hp", "enemy_left_max_hp", "enemy_right_max_hp", "time_ms", "section", "section_type",
               "section_left_ms", "full_bar_ms", "overtime", "play_open_in_ticks", "tick", "episode_step",
               "deck", "deck_slot", "deck_level", "ability_champion", "ability_unit", "ability_state",
               "ability_cooldown_ms", "ability_charges", "ability_command_ticks"]


BOT_INFO = {}
try:
    BOT_INFO = json.loads((HERE / "model" / "bot_info.json").read_text())
except Exception:
    pass
VARIED_LEVELS = bool(BOT_INFO.get("varied_levels"))
VARIED_FORMS = bool(BOT_INFO.get("varied_forms"))
REAL_LEVEL_FIELDS = ["deck_level", "own_king_max_hp", "own_left_max_hp", "own_right_max_hp", "enemy_king_max_hp",
                     "enemy_left_max_hp", "enemy_right_max_hp"]


def view(twin, twin_obs, screen_obs, enc):
    """The observation given to the bot: the twin's, with the hand, elixir, clock and tower state
    read from the screen. For a model trained at level 16 only (it1450), everything is brought back
    to level 16 and to the training forms; for a model trained with varied levels and forms
    (model/bot_info.json), it gets the real ones."""
    if VARIED_LEVELS or VARIED_FORMS:
        return view_real(twin, twin_obs, screen_obs, enc)
    H, E, hi, ei = enc.H, enc.E, enc.header_ints, enc.entity_ints
    ids, names = enc.ids, enc.names
    o = twin_obs.copy()
    for f in FROM_SCREEN:
        for i in H.get(f, []):
            o[i] = screen_obs[i]
    musk, musk_hero = ids.get("Musketeer"), ids.get("MusketeerHero")
    n = int(o[H["entity_count"][0]])
    own_musk = -1
    for k in range(n):
        a = hi + k * ei
        if k < 6:  # towers: HP and state read from the screen
            if twin_obs[a + E["id"]] == screen_obs[a + E["id"]] == k:
                o[a + E["hp"]] = screen_obs[a + E["hp"]]
                o[a + E["max_hp"]] = screen_obs[a + E["max_hp"]]
                o[a + E["flags"]] = (twin_obs[a + E["flags"]] & ~1) | (screen_obs[a + E["flags"]] & 1)
                o[a + E["level"]] = 16
            continue
        if o[a + E["row"]] == musk:  # the bot always saw the Musketeer as a hero
            o[a + E["row"]] = musk_hero
        side = int(o[a + E["side"]])
        if o[a + E["kind"]] == 1:
            mx = int(o[a + E["max_hp"]])
            m16 = twin.hp16.get(int(o[a + E["row"]])) or int(round(mx * twin.k.get(side, 1.0)))
            if mx > 0:
                for f in ("hp", "shield"):
                    if o[a + E[f]] > 0:
                        o[a + E[f]] = int(round(o[a + E[f]] * m16 / mx))
                o[a + E["max_hp"]] = m16
            o[a + E["level"]] = 16
            if side == 0 and o[a + E["row"]] == musk_hero:
                own_musk = int(o[a + E["id"]])
        elif o[a + E["level"]] > 0:
            o[a + E["level"]] = 16
    for c, f in zip(H["enemy_play_cast"], H["enemy_play_form"]):
        if o[c] == musk:
            o[c], o[f] = ids.get("Musketeer_hero", o[c]), 2
    if own_musk >= 0:
        o[H["ability_unit"][0]], o[H["ability_state"][0]] = own_musk, 2
    return o


def add_unseen(o, screen_obs, H, E, header_ints, entity_ints, max_entities=200):
    """Adds the units seen on screen that the twin does not have (an enemy play the vision missed,
    for example): same table row and same side within 2.5 tiles = already there."""
    n_t = int(o[H["entity_count"][0]])
    n_s = int(screen_obs[H["entity_count"][0]])

    def row(buf, k):
        a = header_ints + k * entity_ints
        return buf[a: a + entity_ints]
    twin_rows = [row(o, k) for k in range(6, n_t)]
    added = 0
    for k in range(6, n_s):
        r = row(screen_obs, k)
        if r[E["kind"]] != 1:
            continue
        same = [t for t in twin_rows if t[E["side"]] == r[E["side"]] and t[E["row"]] == r[E["row"]]
                and ((t[E["x"]] - r[E["x"]]) ** 2 + (t[E["y"]] - r[E["y"]]) ** 2) ** 0.5 < 2500]
        if same or n_t >= max_entities:
            continue
        a = header_ints + n_t * entity_ints
        o[a: a + entity_ints] = r
        o[a + E["id"]] = 5000 + int(r[E["id"]])
        n_t += 1
        added += 1
    for i in H["entity_count"]:
        o[i] = n_t
    return o, added


def view_real(twin, twin_obs, screen_obs, enc):
    """The "real levels, real forms" version: the twin's HP and levels as they are; for the towers,
    the screen only gives the share of HP left, applied to the twin's max HP."""
    H, E, hi, ei = enc.H, enc.E, enc.header_ints, enc.entity_ints
    o = twin_obs.copy()
    keep = set(REAL_LEVEL_FIELDS) | ({"deck_slot", "ability_champion", "ability_unit", "ability_state",
                                       "ability_cooldown_ms", "ability_charges", "ability_command_ticks"}
                                      if VARIED_FORMS else set())
    tower_hp = {"own_king_hp": 0, "own_right_hp": 1, "own_left_hp": 2, "enemy_king_hp": 3, "enemy_right_hp": 4,
                "enemy_left_hp": 5}
    for f in FROM_SCREEN:
        if f in keep or f in tower_hp:
            continue
        for i in H.get(f, []):
            o[i] = screen_obs[i]
    for k in range(6):
        a = hi + k * ei
        s_hp, s_max = int(screen_obs[a + E["hp"]]), max(1, int(screen_obs[a + E["max_hp"]]))
        t_max = max(1, int(twin_obs[a + E["max_hp"]]))
        alive = screen_obs[a + E["flags"]] & 1
        o[a + E["hp"]] = int(round(s_hp / s_max * t_max)) if alive else 0
        o[a + E["flags"]] = (twin_obs[a + E["flags"]] & ~1) | alive
    for name, k in tower_hp.items():
        o[H[name][0]] = o[hi + k * ei + E["hp"]]
    return o


# troops the detector sees well: if the screen no longer sees them, they really died
RELIABLE = {"HogRider", "Musketeer", "IceGolemite", "Cannon", "Giant", "Knight", "MiniPekka", "Pekka", "Valkyrie",
            "Golem", "Balloon", "BabyDragon", "Wizard", "Prince", "DarkPrince", "Bowler", "Executioner", "RoyalGiant",
            "MegaKnight", "ElectroWizard", "Witch", "Lumberjack", "InfernoTower", "Tesla", "BombTower", "XBow", "Mortar",
            "LavaHound", "ElixirGolem1", "Sparky", "GoblinGiant", "Miner", "Bandit", "MagicArcher", "Fisherman"}


def family(row):
    return row.replace("_EV1", "").replace("Hero", "").replace("_hero", "")


def drop_ghosts(o, twin, screen_units, now, enc, grace=1.0):
    """Removes from the bot's input the troops the twin thinks are alive but the screen has not seen
    for `grace` s (game randomness or a timing gap made them die in the real match)."""
    H, E, hi, ei, names = enc.H, enc.E, enc.header_ints, enc.entity_ints, enc.names
    seen = [(u["side"], family(u["row"]), u["x"], u["y"]) for u in screen_units if u.get("seen", 3) >= 2]
    n = int(o[H["entity_count"][0]])
    keep, dropped = list(range(6)), 0
    for k in range(6, n):
        a = hi + k * ei
        r = o[a: a + ei]
        name = names[int(r[E["row"]])] if 0 <= r[E["row"]] < len(names) else ""
        fam = family(name)
        if r[E["kind"]] != 1 or fam not in RELIABLE:
            keep.append(k)
            continue
        tid = int(r[E["id"]])
        x, y, side = r[E["x"]] / 1000, r[E["y"]] / 1000, int(r[E["side"]])
        if any(s == side and f == fam and ((ux - x) ** 2 + (uy - y) ** 2) ** 0.5 < 2.5 for s, f, ux, uy in seen):
            twin.last_seen[tid] = now
        twin.last_seen.setdefault(tid, now)
        if now - twin.last_seen[tid] > grace and r[E["state"]] != 4:
            dropped += 1
            continue
        keep.append(k)
    if dropped:
        rows = [o[hi + k * ei: hi + (k + 1) * ei].copy() for k in keep]
        o[hi: hi + n * ei] = 0
        for j, r in enumerate(rows):
            o[hi + j * ei: hi + (j + 1) * ei] = r
        for i in H["entity_count"]:
            o[i] = len(rows)
    if len(twin.last_seen) > 2000:
        twin.last_seen = {k: v for k, v in twin.last_seen.items() if now - v < 10}
    return o, dropped
