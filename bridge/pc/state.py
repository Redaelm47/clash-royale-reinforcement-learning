"""What the bot knows about the match, updated frame after frame, and the crforge observation built from it.

The bot learned in crforge with a very rich observation. Here it is rebuilt field by field, in the
same format (read from a real crforge match on Nov 18):
- your elixir: the bar (number of pink segments), corrected right after a play (the bar empties
  with an animation and is wrong for ~1 s), with the measured rate (x1, x2);
- your hand and your cycle: the played card goes to the end of the queue, the next one takes its
  place. The image only confirms it, and a card that is not recognized (purple evolution frame,
  greyed-out card) is found by elimination;
- the pending card during the first second after a play, the reserved elixir, the hand refill
  delay, the evolution counter of the Cannon and the Skeletons;
- the enemy: each new unit seen becomes an enemy play (card, point, time), which gives the seen
  cards, the last 8 plays and the enemy elixir counted the way a player counts it;
- the towers: up or down, and their HP read from their health bar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INFO = json.loads((HERE / "model" / "cards_info.json").read_text())
MASKS = json.loads((HERE / "model" / "masks.json").read_text())

# The 2.6 deck, in the order shown when labelling cards: (table name, display name, cost, type)
DECK = [
    ("HogRider", "Hog Rider", 4, "troop"),
    ("Musketeer", "Musketeer", 4, "troop"),
    ("Cannon", "Cannon", 3, "building"),
    ("IceGolemite", "Ice Golem", 2, "troop"),
    ("IceSpirits", "Ice Spirit", 1, "troop"),
    ("Skeletons", "Skeletons", 1, "troop"),
    ("Fireball", "Fireball", 4, "spell"),
    ("Log", "The Log", 2, "spell"),
]
DECK_NAMES = [d[0] for d in DECK]
BY_NAME = {d[0]: d for d in DECK}
# As in training: Musketeer as a hero (2), Cannon and Skeletons as evolutions (1)
SLOT_FLAGS = [0, 2, 1, 0, 0, 1, 0, 0]
EVO_NEED = {"Cannon": 2, "Skeletons": 2}
HERO = {"Musketeer"}
CAST = {("Musketeer", 2): "Musketeer_hero", ("Skeletons", 1): "Skeletons_EV1", ("Cannon", 1): "Cannon_EV1"}

# Towers in crforge order (ids 0 to 5): (name, side, x, y, max HP at level 16)
TOWERS = [
    ("KingTower", 0, 9.0, 3.0, 7728), ("PrincessTower", 0, 14.5, 6.5, 4858), ("PrincessTower", 0, 3.5, 6.5, 4858),
    ("KingTower", 1, 9.0, 29.0, 7728), ("PrincessTower", 1, 14.5, 25.5, 4858), ("PrincessTower", 1, 3.5, 25.5, 4858),
]
HEADER_TOWERS = {"own_king": 0, "own_right": 1, "own_left": 2, "enemy_king": 3, "enemy_right": 4, "enemy_left": 5}

UNIT_HP = dict(INFO["unit_hp"])
UNIT_HP.update({"MusketeerHero": 1153, "Skeleton_EV1": 130, "Cannon_EV1": 1316})
UNIT_TO_CARD = dict(INFO["unit_to_card"])
UNIT_TO_CARD.update({"MusketeerHero": "Musketeer", "Skeleton_EV1": "Skeletons", "Cannon_EV1": "Cannon",
                     "FireballSpell": "Fireball", "LogProjectileRolling": "Log", "Skeleton": "Skeletons"})
CARD_COST = {name: c["cost"] for name, c in INFO["cards"].items()}
CARD_COST.update({d[0]: d[2] for d in DECK})


DECK26_CLASSES = {"hog-rider", "musketeer", "cannon", "cannon-cart", "ice-golem", "ice-spirit", "ice-spirit-evolution",
                  "skeleton", "skeleton-evolution", "fireball", "the-log"}


def configure_forms(evolutions, heroes):
    """For a model trained with varied forms: the deck as it really is on the account (instead of
    the hero Musketeer and the evolved Cannon and Skeletons used in training)."""
    for i, c in enumerate(DECK_NAMES):
        SLOT_FLAGS[i] = 2 if c in heroes else (1 if c in evolutions else 0)
    EVO_NEED.clear()
    EVO_NEED.update({c: 2 for c in evolutions})
    HERO.clear()
    HERO.update(heroes)


def grid(rows):
    return np.array([[ch == "1" for ch in r] for r in rows], dtype=bool)


BASE_MASK = {c: grid(r) for c, r in MASKS["base"].items()}
POCKET_L = {c: grid(r) for c, r in MASKS["pocket_left"].items()}
POCKET_R = {c: grid(r) for c, r in MASKS["pocket_right"].items()}


def refill_cooldown(t):
    return 1.0 if t < 120 else (0.5 if t < 240 else 0.35)


# -- your elixir ------------------------------------------------------------------------------------

class Elixir:
    """Your elixir (a float, in elixir). The bar gives the integer part; the fraction follows the regen."""

    def __init__(self):
        self.e = None
        self.t = None
        self.last_bar = None
        self.hold_until = 0.0
        self.incs = []      # times when the bar gained a segment (outside plays)
        self.mult = 1       # 1 normal, 2 double elixir
        self.mult_changed_at = None
        self.stable = 0     # number of identical reads in a row (a gained segment only counts after a stable bar)

    def update(self, bar, now):
        if self.e is None:
            if bar is not None:
                self.e, self.t, self.last_bar = bar + 0.5, now, bar
            return
        pred = min(10.0, self.e + (now - self.t) * self.mult / 2.8)
        self.t = now
        if bar is not None:
            if now < self.hold_until:
                # right after your play the bar still shows the old total while it empties,
                # so we only trust it when it says less than expected
                if bar < int(pred):
                    pred = bar + 0.98
            else:
                if self.last_bar is not None and bar == self.last_bar + 1 and bar < 10 and self.stable >= 2:
                    self.incs.append(now)
                    self._rate()
                self.stable = self.stable + 1 if bar == self.last_bar else 0
                if pred < bar:
                    pred = bar + 0.02
                elif pred >= bar + 1:
                    pred = bar + 0.98
                self.last_bar = bar
        self.e = max(0.0, min(10.0, pred))

    def _rate(self):
        """x1, x2 or x3 from the last two gaps between segments gained in a row. The rate only goes
        up during a match."""
        if len(self.incs) < 3:
            return
        d = np.diff(self.incs[-3:])
        m = 3 if d.max() < 1.15 else (2 if d.max() < 2.0 else 1)
        if m > self.mult:
            self.mult, self.mult_changed_at = m, self.incs[-3] - 0.5

    def on_play(self, cost, now):
        if self.e is not None:
            self.e = max(0.0, self.e - cost)
        self.hold_until = now + 1.6
        self.last_bar = None
        self.stable = 0


# -- your hand and your cycle ------------------------------------------------------------------------

class Hand:
    def __init__(self):
        self.hand = [None] * 4
        self.queue = [None] * 4           # behind the hand, in the order the cards come back
        self.prev = [None] * 4
        self.prev_next = None
        self.pending = {}                 # slot -> play given less than 1 s ago (crforge keeps it "pending")
        self.busy_until = [0.0] * 4       # slot animating after a play: don't trust the image
        self.empty_until = [0.0] * 4
        self.check = {}                   # slot -> (played card, until when we check that it left)
        self.last_refill = -10.0
        self.evo = {c: 0 for c in EVO_NEED}
        self.evo_live = {}                # card -> time of its last evolved play
        self.failed = 0
        self.on_revert = None             # called with the card of a refused play (for the twin)

    def form(self, card):
        if card in HERO:
            return 2
        if card in EVO_NEED and self.evo[card] >= EVO_NEED[card]:
            return 1
        return 0

    def _set(self, k, name, sure=()):
        if self.hand[k] == name:
            return
        for j in range(4):
            if j != k and self.hand[j] == name and j not in sure:
                self.hand[j] = None
        self.queue = [None if q == name else q for q in self.queue]
        self.hand[k] = name

    def _solve(self):
        known = {h for h in self.hand if h} | {q for q in self.queue if q}
        missing = [c for c in DECK_NAMES if c not in known]
        holes = [("h", k) for k in range(4) if self.hand[k] is None] + \
                [("q", i) for i in range(4) if self.queue[i] is None]
        if len(missing) == 1 and len(holes) == 1:
            kind, i = holes[0]
            if kind == "h":
                self.hand[i] = missing[0]
            else:
                self.queue[i] = missing[0]

    def update(self, reads, next_read, now, t_game, empties=None):
        # plays whose one-second wait is over go into play
        for k in [k for k, p in self.pending.items() if now >= p["t"] + 1.0]:
            self._run(k, self.pending.pop(k), t_game)
        # a slot that looks empty on screen is empty: the next card has not arrived yet (the real game
        # takes about 2 s after the play, crforge 1 s less). Without this the bot dragged from empty slots.
        for k, empty in enumerate(empties or ()):
            if empty and k not in self.pending:
                self.empty_until[k] = max(self.empty_until[k], now + 0.3)
        sure = set()
        for k, (name, score) in enumerate(reads):
            ok = name is not None and (score >= 0.88 or (score >= 0.7 and self.prev[k] == name))
            self.prev[k] = name if (name is not None and score >= 0.7) else None
            if k in self.pending or now < self.busy_until[k]:
                continue
            if k in self.check:
                card, until = self.check[k]
                if now > until:
                    del self.check[k]
                elif ok and name == card:
                    self._revert(k, card)
                    continue
            if ok:
                self._set(k, name, sure)
                sure.add(k)
        name, score = next_read
        ok = name is not None and (score >= 0.85 or (score >= 0.65 and self.prev_next == name))
        self.prev_next = name if (name is not None and score >= 0.65) else None
        if ok and name not in {self.hand[k] for k in sure}:
            for j in range(4):
                if self.hand[j] == name and j not in sure:
                    self.hand[j] = None
            self.queue = [None if q == name else q for q in self.queue]
            self.queue[0] = name
        self._solve()

    def play(self, k, card, tile, now):
        self.pending[k] = {"card": card, "t": now, "tile": tile, "form": self.form(card)}

    def _run(self, k, p, t_game):
        """The play "runs" (1 s later), as in crforge: the next card takes the slot."""
        card = p["card"]
        nxt = self.queue[0]
        self.queue = self.queue[1:] + [card]
        cd = refill_cooldown(t_game)
        t_run = p["t"] + 1.0
        if t_run - self.last_refill >= cd:
            self.last_refill = t_run
        else:
            self.last_refill += cd
            self.empty_until[k] = self.last_refill
        self.hand[k] = nxt
        self.busy_until[k] = t_run + 0.5
        self.check[k] = (card, t_run + 2.5)
        p["prev_evo"] = self.evo.get(card)
        self.last_run = p
        if card in EVO_NEED:
            if p["form"] == 1:
                self.evo[card] = 0
                self.evo_live[card] = t_run
            else:
                self.evo[card] = min(EVO_NEED[card], self.evo[card] + 1)
        self._solve()

    def _revert(self, k, card):
        """The card came back to its slot: the play did not go through (elixir, tile refused...)."""
        self.failed += 1
        self.check.pop(k, None)
        if self.on_revert:
            self.on_revert(card)
        back = self.hand[k]
        if self.queue and self.queue[-1] == card:
            self.queue = [back] + self.queue[:-1]
        self.hand[k] = card
        if card in EVO_NEED and getattr(self, "last_run", {}).get("card") == card:
            prev = self.last_run.get("prev_evo")
            if prev is not None:
                self.evo[card] = prev

    def view(self, now):
        """The hand as crforge would show it: (cards, pending)."""
        cards, pend = [], []
        for k in range(4):
            if k in self.pending:
                cards.append(self.pending[k]["card"])
                pend.append(1)
            elif now < self.empty_until[k]:
                cards.append(None)
                pend.append(0)
            else:
                cards.append(self.hand[k])
                pend.append(0)
        return cards, pend


# -- the enemy ----------------------------------------------------------------------------------------

class Enemy:
    def __init__(self):
        self.plays = []        # oldest to newest
        self.seen = []
        self.evo = {}
        self.elixir = None
        self.count = 0
        self.done = set()      # track ids already counted as a play
        self.last = {}         # id -> unit seen in the previous frame
        self.lost = []         # (name, x, y, t) of units lost from view recently
        self.fireballs = {}    # track id -> play (the point is only known on impact)
        self.on_play = None    # called with (card, time it appeared, x, y, form) for the twin
        from vision import Clocks
        self.clocks = Clocks()  # deploy clocks: exact point and time of plays
        self.from_clock = 0
        self.orphans = []      # enemy clocks with no recognized troop (plays of an unknown card)
        self.on_resolve = None  # called when a troop later identifies one of these plays

    def update(self, units, now, mult, towers_alive, clocks=()):
        self.clocks.update(clocks, now)
        # an enemy clock still without a recognized troop: a play of an unknown card, at the exact
        # point and time (the twin will guess the card from the cycle)
        for c in self.clocks.items:
            if c["side"] == 1 and not c["used"] and now - c["t"] > 2.5 and c["y"] >= 14.5:
                c["used"] = True
                self.orphans.append(c)
                if self.on_play:
                    self.on_play(None, c["t"], c["x"], c["y"], 0)
        if self.elixir is not None:
            self.elixir = min(10.0, self.elixir + (now - getattr(self, "t", now)) * mult / 2.8)
        self.t = now
        cur = {u["id"]: u for u in units if u["side"] == 1}
        for tid, u in self.last.items():
            if tid not in cur:
                self.lost.append((u["name"], u["x"], u["y"], now))
                if tid in self.fireballs:  # the Fireball hit: its point is where we saw it last
                    p = self.fireballs.pop(tid)
                    p["x"], p["y"], p["t_point"], p["flags"] = u["x"], u["y"], now, p["flags"] | 2
                    if self.on_play:
                        self.on_play(p["card"], p["t"], p["x"], p["y"], p["form"])
        self.lost = [l for l in self.lost if now - l[3] < 3.0]
        for tid, u in cur.items():
            # a card outside the 2.6 must be seen in 3 frames (false detections from the detector)
            if tid in self.done or u["seen"] < (2 if u["name"] in DECK26_CLASSES else 3):
                continue
            self.done.add(tid)
            card = UNIT_TO_CARD.get(u["row"])
            if card is None:
                continue
            name = u["name"]
            spell = name in ("fireball", "the-log", "zap", "arrows", "rocket", "poison", "lightning",
                             "earthquake", "giant-snowball", "tornado", "freeze", "rage", "barbarian-barrel")
            x0, y0, t_play = u["x0"], u["y0"], u["born"]
            clock = None
            if not spell:
                # the closest deploy clock gives the exact point and time
                clock = self.clocks.match(1, x0, y0, u["born"])
                if clock is not None:
                    x0, y0, t_play = clock["x"], clock["y"], clock["t"]
                # a troop can only be placed on their side (or in a pocket if one of your towers is down)
                pocket = (x0 < 9 and not towers_alive[2]) or (x0 >= 9 and not towers_alive[1])
                if y0 < 15.0 and not (pocket and y0 >= 8.0):
                    continue
                # a unit lost and found again is not a new play (unless a new clock confirms it)
                if clock is None and any(n == name and abs(lx - x0) < 4 and abs(ly - y0) < 4
                                         for n, lx, ly, _ in self.lost):
                    continue
            # several units from the same card (Skeletons, Spirits...) = a single play
            same = [p for p in self.plays if p["card"] == card and abs(t_play - p["t"]) < 1.5
                    and p["x"] is not None and abs(p["x"] - x0) < 4 and abs(p["y"] - y0) < 4]
            if same:
                continue
            if clock is not None:
                clock["used"] = True
                self.from_clock += 1
            # the troop of a clock already counted as an unknown-card play: give the card to that play
            orphan = next((o for o in self.orphans if abs(o["x"] - x0) < 3 and abs(o["y"] - y0) < 3
                           and -1.0 < t_play - o["t"] < 3.0), None)
            if spell:
                orphan = None
            if orphan is not None:
                self.orphans.remove(orphan)
                if self.on_resolve:
                    self.on_resolve(card, orphan["t"], orphan["x"], orphan["y"])
                x0, y0, t_play = orphan["x"], orphan["y"], orphan["t"]
            form = 2 if card == "Musketeer" else (1 if u["row"] in ("Skeleton_EV1", "Cannon_EV1") else 0)
            cast = CAST.get((card, form), card)
            kind = 3 if name in ("fireball", "rocket", "arrows", "giant-snowball") else \
                (4 if name in ("the-log", "barbarian-barrel") else (2 if spell else 1))
            flags = 1 | (2 if kind != 3 else 0) | (8 if card in EVO_NEED and form == 0 else 0)
            p = {"card": card, "cast": cast, "form": form, "t": t_play, "x": x0, "y": y0, "kind": kind,
                 "flags": flags, "t_point": t_play if kind != 3 else None}
            if kind == 3:
                p["x"] = p["y"] = None
                self.fireballs[tid] = p
            elif self.on_play and orphan is None:
                self.on_play(card, p["t"], x0, y0, form)
            self.plays.append(p)
            self.count += 1
            if card not in self.seen and len(self.seen) < 8:
                self.seen.append(card)
            if card in EVO_NEED:
                self.evo[card] = 0 if form == 1 else min(EVO_NEED[card], self.evo.get(card, 0) + 1)
            if self.elixir is not None:
                self.elixir = max(0.0, self.elixir - CARD_COST.get(cast, CARD_COST.get(card, 3)))
        self.last = cur


# -- the towers ---------------------------------------------------------------------------------------

class Towers:
    def __init__(self, now):
        self.seen = [now] * 6
        self.miss = [0] * 6
        self.alive = [True] * 6
        self.hp = [1.0] * 6          # as a fraction of max HP (0 in the observation if the tower is down)
        self.cand = [None] * 6

    def update(self, boxes, bars, now):
        """boxes: (side, x, y) of the detected towers; bars: (side, type, x, y, fill)."""
        for k, (name, side, x, y, _) in enumerate(TOWERS):
            if any(s == side and abs(tx - x) < 2.5 and abs(ty - y) < 3.5 for s, tx, ty in boxes):
                self.seen[k], self.miss[k] = now, 0
            else:
                self.miss[k] += 1
            # down: not seen for 4 s and in at least 6 frames in a row (a troop can hide it)
            self.alive[k] = not (now - self.seen[k] >= 4.0 and self.miss[k] >= 6)
        for side, kind, bx, by, f in bars:
            if f is None:
                continue
            cands = [k for k, t in enumerate(TOWERS) if t[1] == side and (t[0] == "KingTower") == (kind == "king-tower-bar")]
            if not cands:
                continue
            k = min(cands, key=lambda k: abs(TOWERS[k][2] - bx))
            if abs(TOWERS[k][2] - bx) > 3 or not self.alive[k]:
                continue
            # HP never goes up, and a read is only taken if the previous frame said the same
            if f <= self.hp[k] + 0.03 and f >= self.hp[k] - 0.5:
                if self.cand[k] is not None and abs(self.cand[k] - f) < 0.06:
                    self.hp[k] = min(self.hp[k], f)
                self.cand[k] = f
            else:
                self.cand[k] = None

    def awake(self, side):
        """The King is awake once it has taken damage or one of its Princess Towers is down."""
        b = 0 if side == 0 else 3
        return self.hp[b] < 0.97 or not self.alive[b + 1] or not self.alive[b + 2]

    def frac(self, k):
        return self.hp[k] if self.alive[k] else 0.0


# -- the match ----------------------------------------------------------------------------------------

class Game:
    def __init__(self, now, twin=None):
        self.t0 = now
        self.twin = twin
        self.played = False      # at least one play made (after that, elixir no longer tells the time)
        self.start_fixed = False
        self.elixir = Elixir()
        self.hand = Hand()
        self.enemy = Enemy()
        self.towers = Towers(now)
        self.units = []
        self.time_fixed = False
        self.clock_synced = False   # clock synced to the on-screen timer
        self.timer_prev = None
        self.timer_cand = None
        self.timer_phase, self.ot_len = 0, 60
        self.own_fireball_aim = None
        self.geo_ready = True   # false until the grid is aligned on the towers
        self.recent = []        # your last plays, to check from the elixir that they went through
        if twin is not None:
            self.hand.on_revert = twin.remove_own
            self.enemy.on_play = twin.add_enemy
            self.enemy.on_resolve = twin.resolve

    def t(self, now):
        return now - self.t0

    def _sync_clock(self, r, now):
        """Syncs the clock to the on-screen timer (r seconds shown). The game rounds the time left up to
        the next second: when the display goes from r+1 to r, exactly r s are left."""
        prev, self.timer_prev = self.timer_prev, (r, now)
        if prev is not None and self.timer_phase == 0 and prev[0] <= 10 and r > prev[0] + 10:
            self.timer_phase, self.ot_len = 1, 60 * int(np.ceil(r / 60))  # overtime: the timer starts again
        elapsed = (180 - r) if self.timer_phase == 0 else 180 + (self.ot_len - r)
        exact = prev is not None and prev[0] == r + 1 and now - prev[1] <= 0.6
        t0 = (prev[1] + now) / 2 - elapsed if exact else now - (elapsed + 0.5)
        cand, self.timer_cand = self.timer_cand, t0
        if cand is None or abs(cand - t0) > 1.2:  # two reads must agree before we touch the clock
            return
        if not self.clock_synced or (exact and abs(t0 - self.t0) > 0.3) or abs(t0 - self.t0) > 1.2:
            self.t0 = t0
            self.clock_synced = self.start_fixed = self.time_fixed = True
        if self.timer_phase == 1 or elapsed >= 120:
            self.elixir.mult = max(self.elixir.mult, 2)

    def update(self, read, now):
        if read.get("timer") is not None:
            self._sync_clock(read["timer"], now)
        t = self.t(now)
        n_inc = len(self.elixir.incs)
        self.elixir.update(read.get("elixir"), now)
        # before your first play the elixir tells the time: the match starts at 6 and gains 1 every 2.8 s
        if not self.played and not self.start_fixed and len(self.elixir.incs) > n_inc:
            bar = read.get("elixir")
            if bar is not None and 7 <= bar <= 9:
                t_real = (bar - 6) * 2.8
                if abs((now - t_real) - self.t0) > 1.0:
                    self.t0 = now - t_real
                self.start_fixed = True
        # double elixir starts with 2:00 left (120 s into the match): if it comes early or late, fix the clock
        e = self.elixir
        if not self.time_fixed and e.mult == 2 and e.mult_changed_at is not None:
            tc = e.mult_changed_at - self.t0
            if abs(tc - 120) < 30:
                self.t0 += tc - 120
            self.time_fixed = True
        if self.enemy.elixir is None and self.elixir.e is not None:
            self.enemy.elixir = self.elixir.e
        self.hand.update(read["hand"], read["next"], now, self.t(now), read.get("empty"))
        self._check_plays(read.get("elixir"), now)
        if "units" in read:
            self.towers.update(read["tower_boxes"], read["tower_bars"], now)
            self.units = read["units"]
            self.enemy.update(self.units, now, self.elixir.mult, self.towers.alive, read.get("clocks", ()))

    def update_vision(self, units, boxes, bars, clocks, now):
        """One more frame between two decisions: only the troops, the towers and the enemy plays."""
        self.towers.update(boxes, bars, now)
        self.units = units
        self.enemy.update(units, now, self.elixir.mult, self.towers.alive, clocks)

    def _check_plays(self, bar, now):
        """2.3 s after a play, the bar must have dropped by its cost. Otherwise the game refused it
        (useful for cards the image does not recognize, like the evolved Skeletons)."""
        if bar is None:
            return
        for pc in self.recent:
            if pc["done"] or now - pc["t"] < 2.3:
                continue
            pc["done"] = True
            if now - pc["t"] > 4.0 or pc["e0"] is None or any(o["t"] > pc["t"] for o in self.recent):
                continue
            fail_e = min(10.0, pc["e0"] + (now - pc["t"]) * self.elixir.mult / 2.8)
            ok_e = max(0.0, fail_e - pc["cost"])
            if abs(bar - fail_e) + 0.5 < abs(bar - ok_e):
                ck = self.hand.check.get(pc["slot"])
                if ck and ck[0] == pc["card"]:
                    self.hand._revert(pc["slot"], pc["card"])
                self.elixir.e = bar + 0.5
        self.recent = [pc for pc in self.recent if now - pc["t"] < 5.0]

    def on_play(self, slot, card, tile, now):
        self.played = True
        if self.twin is not None:
            self.twin.add_own(card, now, tile)
        self.recent.append({"slot": slot, "card": card, "t": now, "e0": self.elixir.e,
                            "cost": BY_NAME[card][2], "done": False})
        self.hand.play(slot, card, tile, now)
        self.elixir.on_play(BY_NAME[card][2], now)
        if card == "Fireball":
            self.own_fireball_aim = (tile[0] + 0.5, tile[1] + 0.5, now)

    # ---- action mask ----
    def mask(self, actions, now):
        m = np.zeros(actions, dtype=bool)
        m[0] = True
        t = self.t(now)
        if t < 4.5 or self.elixir.e is None or not self.geo_ready:  # crforge: no card during the first 90 ticks
            return m
        cards, pend = self.hand.view(now)
        avail = int(self.elixir.e + 1e-6)
        left_open, right_open = not self.towers.alive[5], not self.towers.alive[4]
        for k, c in enumerate(cards):
            if not c or pend[k] or k in self.hand.pending or avail < BY_NAME[c][2]:
                continue
            g = BASE_MASK[c].copy()
            if left_open:
                g |= POCKET_L[c]
            if right_open:
                g |= POCKET_R[c]
            m[1 + k * 576: 1 + (k + 1) * 576] = g.reshape(-1)
        return m

    # ---- crforge observation ----
    def observation(self, enc, now):
        o = np.zeros(enc.seat_ints, dtype=np.int32)
        put = enc.putter(o)
        ids = enc.ids
        t = self.t(now)
        ms = int(t * 1000)
        cards, pend = self.hand.view(now)
        e = self.elixir.e if self.elixir.e is not None else 5.0
        reserved = sum(BY_NAME[p["card"]][2] for p in self.hand.pending.values())
        mult = self.elixir.mult
        put("tick", int(t * 20))
        put("episode_step", int(t * 4))
        put("elixir", int((e + reserved) * 10000))
        put("elixir_reserved", reserved * 10000)
        dead = [not a for a in self.towers.alive]
        put("crowns", (3 if dead[3] else dead[4] + dead[5]))
        put("crowns_enemy", (3 if dead[0] else dead[1] + dead[2]))
        put("hand_refill_ms", int(max(0.0, self.hand.last_refill + refill_cooldown(t) - now) * 1000))
        put("hand_card", [ids[c] if c else -1 for c in cards])
        put("hand_cost", [BY_NAME[c][2] if c else 0 for c in cards])
        put("hand_pending", pend)
        q = self.hand.queue
        put("next_card", ids[q[0]] if q[0] else -1)
        put("queue", [ids[c] if c else -1 for c in q])
        put("deck", [ids[c] for c in DECK_NAMES])
        put("deck_level", [16] * 8)
        put("deck_slot", SLOT_FLAGS)
        put("deck_evolution", [self.hand.evo.get(c, 0) for c in DECK_NAMES])
        put("hand_form", [self.hand.form(c) if c else 0 for c in cards])
        for key, k in HEADER_TOWERS.items():
            mx = TOWERS[k][4]
            put(key + "_hp", int(round(self.towers.hp[k] * mx)) if self.towers.alive[k] else 0)
            put(key + "_max_hp", mx)
        put("decision_ticks", 5)
        put("play_delay_ticks", 20)
        put("seat", 0)
        put("time_ms", ms)
        overtime = t >= 180
        put("section", 1 if overtime else 0)
        put("section_type", 1 if overtime else 0)
        put("section_left_ms", int(max(0, (300 - t) if overtime else (180 - t)) * 1000))
        put("full_bar_ms", 28000 if mult == 1 and t < 120 else 14000)
        put("overtime", int(overtime))
        put("play_open_in_ticks", max(0, int(90 - t * 20)))
        # your pending plays (crforge holds them 20 ticks before placing them)
        pend_rows = sorted(self.hand.pending.values(), key=lambda p: p["t"])
        put("pending_count", len(pend_rows))
        put("pending_card", [ids[p["card"]] for p in pend_rows] + [-1] * (4 - len(pend_rows)))
        put("pending_x", [int((p["tile"][0] + 0.5) * 1000) for p in pend_rows] + [0] * (4 - len(pend_rows)))
        put("pending_y", [int((p["tile"][1] + 0.5) * 1000) for p in pend_rows] + [0] * (4 - len(pend_rows)))
        put("pending_ticks", [max(0, int(20 - (now - p["t"]) * 20)) for p in pend_rows] + [0] * (4 - len(pend_rows)))
        # the hero Musketeer from training (its ability is never used here)
        musk = [u for u in self.units if u["side"] == 0 and u["row"] == "MusketeerHero"]
        played_musk = any(p == "Musketeer" for p in self.hand.queue) or bool(musk)
        put("ability_champion", [ids["MusketeerHero"], -1])
        put("ability_unit", [musk[0]["id"] if musk else -1, -1])
        put("ability_state", [2 if musk else 1, 0])
        put("ability_cooldown_ms", [0, 0])
        put("ability_charges", [1 if played_musk else -1, -1])
        put("ability_command_ticks", [0, 0])
        # the enemy
        en = self.enemy
        put("enemy_elixir", int((en.elixir if en.elixir is not None else e) * 10000))
        put("enemy_play_count", en.count)
        put("enemy_seen", [ids.get(c, -1) for c in en.seen] + [-1] * (8 - len(en.seen)))
        put("enemy_seen_evolution", [(en.evo[c] if c in en.evo else -1) for c in en.seen] + [-1] * (8 - len(en.seen)))
        put("enemy_seen_evolution_need", [EVO_NEED.get(c, 0) if c in en.evo else 0 for c in en.seen] + [0] * (8 - len(en.seen)))
        rows = list(reversed(en.plays[-8:]))
        f = {k: [] for k in ("card", "cast", "form", "ticks", "x", "y", "kind", "flags", "point")}
        for p in rows:
            f["card"].append(ids.get(p["card"], -1))
            f["cast"].append(ids.get(p["cast"], -1))
            f["form"].append(p["form"])
            f["ticks"].append(max(1, int((now - p["t"]) * 20)))
            f["x"].append(int(p["x"] * 1000) if p["x"] is not None else -1)
            f["y"].append(int(p["y"] * 1000) if p["y"] is not None else -1)
            f["kind"].append(p["kind"])
            f["flags"].append(p["flags"])
            f["point"].append(max(1, int((now - p["t_point"]) * 20)) if p["t_point"] is not None else -1)
        pad = 8 - len(rows)
        put("enemy_play_card", f["card"] + [-1] * pad)
        put("enemy_play_cast", f["cast"] + [-1] * pad)
        put("enemy_play_form", f["form"] + [0] * pad)
        put("enemy_play_ticks", f["ticks"] + [0] * pad)
        put("enemy_play_x", f["x"] + [0] * pad)
        put("enemy_play_y", f["y"] + [0] * pad)
        put("enemy_play_kind", f["kind"] + [0] * pad)
        put("enemy_play_flags", f["flags"] + [0] * pad)
        put("enemy_play_point_ticks", f["point"] + [0] * pad)
        put("ability_champion", [ids["MusketeerHero"], -1])
        # entities: the 6 towers, then the units seen
        n = 0
        for k, (name, side, x, y, mx) in enumerate(TOWERS):
            up = self.towers.alive[k]
            flags = (1 if up else 0) | (1 << 5)
            if name == "KingTower":
                flags |= (1 << 6) | ((1 << 8) if self.towers.awake(side) else 0)
            else:
                flags |= 1 << 8
            enc.entity(o, n, id=k, kind=2, side=side, row=ids[name], x=int(x * 1000), y=int(y * 1000),
                       hp=int(round(self.towers.hp[k] * mx)) if up else 0, max_hp=mx, state=0, flags=flags,
                       level=16, target=-1, speed_percent=100, hit_speed_percent=100, buff_0=-1, buff_1=-1,
                       buff_2=-1, ability_ticks=-1)
            n += 1
        evo_cannon = now - self.hand.evo_live.get("Cannon", -1e9) < 35
        for u in sorted(self.units, key=lambda u: u["id"]):
            if n >= 60 or u["row"] not in ids or u.get("seen", 3) < 2:  # seen at least twice (false detections)
                continue
            row = u["row"]
            if u["side"] == 0 and row == "Cannon" and evo_cannon:
                row = "Cannon_EV1"
            common = dict(id=u["id"], side=u["side"], row=ids[row], x=int(u["x"] * 1000), y=int(u["y"] * 1000),
                          dx=int(u["dx"] * 1000), dy=int(u["dy"] * 1000), level=16, target=-1, buff_0=-1,
                          buff_1=-1, buff_2=-1, ability_ticks=-1)
            if u["kind"] == 1:
                mx = UNIT_HP.get(row, UNIT_HP.get(u["row"], 0))
                age = now - u["born"]
                moving = abs(u["dx"]) + abs(u["dy"]) > 0.05
                state = 4 if age < 0.8 else (0 if u["building"] else (1 if moving else 2))
                hp = mx
                if u["side"] == 1 and (row == "MusketeerHero" or state == 4):
                    hp = -1  # crforge: enemy HP is hidden until the bar is drawn
                enc.entity(o, n, kind=1, hp=hp, max_hp=mx, state=state,
                           flags=1 | ((1 << 4) if u["air"] else 0) | ((1 << 5) if u["building"] else 0),
                           deploy_ms=int(max(0.0, 1.0 - age) * 1000) if state == 4 else 0,
                           speed_percent=100, hit_speed_percent=100, **common)
            else:
                aim = (0, 0)
                if row == "LogProjectileRolling":
                    aim = (int(u["x"] * 1000), int((u["y"] + (3 if u["side"] == 0 else -3)) * 1000))
                elif row == "FireballSpell":
                    if u["side"] == 1:
                        aim = (-1, -1)
                    elif self.own_fireball_aim and now - self.own_fireball_aim[2] < 4:
                        aim = (int(self.own_fireball_aim[0] * 1000), int(self.own_fireball_aim[1] * 1000))
                enc.entity(o, n, kind=u["kind"], hp=0, max_hp=0, state=-1, flags=1, aim_x=aim[0], aim_y=aim[1],
                           **common)
            n += 1
        put("entity_count", n)
        return o


class Encoder:
    """The positions of the fields in the observation of 7,024 ints (from metadata.json)."""

    def __init__(self, meta):
        sch = meta["schema"]
        self.H = {}
        for i, f in enumerate(sch["header_fields"]):
            self.H.setdefault(f, []).append(i)
        self.E = {f: i for i, f in enumerate(sch["entity_fields"])}
        self.names = meta["vocabulary"]["names"]
        self.ids = {n: i for i, n in enumerate(self.names)}
        self.header_ints, self.entity_ints = int(sch["header_ints"]), int(sch["entity_ints"])
        self.seat_ints, self.actions = int(sch["seat_ints"]), int(sch["actions"])

    def putter(self, o):
        def put(name, values):
            idx = self.H.get(name)
            if idx is None:
                return
            values = values if isinstance(values, (list, tuple)) else [values]
            for i, v in zip(idx, values):
                o[i] = int(v)
        return put

    def entity(self, o, n, **fields):
        base = self.header_ints + n * self.entity_ints
        for k, v in fields.items():
            if k in self.E:
                o[base + self.E[k]] = int(v)
