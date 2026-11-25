"""Unit vision: the two YOLOv8 detectors from KataCR (MIT license, github.com/wty-yy/KataCR).

The arena is cropped from the capture the way KataCR did it (a 576 x 896 image where the 18 x 32
grid fills the rectangle (4, 82) -> (574, 884)), units are detected, then converted to tiles in
the player's frame (x to the right, y from their back line).

Only the model definition is taken from KataCR (vendor/katacr/yolov8/custom_model.py), so the
weights can be loaded. The weight files go in model/:
detector1_v0.7.13.pt and detector2_v0.7.13.pt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

BG_W, BG_H = 576, 896
GRID = (4, 82, 574, 884)
CELL_W = (GRID[2] - GRID[0]) / 18
CELL_H = (GRID[3] - GRID[1]) / 32
WEIGHTS = [HERE / "model" / "detector1_v0.7.13.pt", HERE / "model" / "detector2_v0.7.13.pt"]

# Your units: only the ones in your deck. The detector sometimes mistakes the Cannon for the
# Cannon Cart. The Musketeer becomes "MusketeerHero", the form the bot always saw in training
# (crforge 2.6 deck with the Musketeer as a hero).
OWN = {
    "hog-rider": "HogRider", "musketeer": "MusketeerHero", "cannon": "Cannon", "cannon-cart": "Cannon",
    "ice-golem": "IceGolemite", "ice-spirit": "IceSpirits", "ice-spirit-evolution": "IceSpirits",
    "skeleton": "Skeleton", "skeleton-evolution": "Skeleton_EV1", "fireball": "FireballSpell",
    "the-log": "LogProjectileRolling",
}
# Enemy units: KataCR names -> crforge table rows
SPECIAL = {
    "hog-rider": "HogRider", "musketeer": "MusketeerHero", "cannon": "Cannon", "cannon-cart": "Cannon",
    "ice-golem": "IceGolemite", "ice-spirit": "IceSpirits", "ice-spirit-evolution": "IceSpirits",
    "skeleton": "Skeleton", "skeleton-evolution": "Skeleton_EV1", "fireball": "FireballSpell",
    "the-log": "LogProjectileRolling", "archer": "Archer", "goblin": "Goblin", "spear-goblin": "SpearGoblin",
    "barbarian": "Barbarian", "minion": "Minion", "bat": "Bat", "golemite": "Golemite", "lava-pup": "LavaPups",
    "royal-hog": "RoyalHog", "elixir-golem-big": "ElixirGolem1", "elixir-golem-mid": "ElixirGolem2",
    "elixir-golem-small": "ElixirGolem4",
}
# The 2.6 classes: on the enemy side, any other class needs a higher confidence to be kept (false
# detections seen on the captures: Archer Queen, Sparky, Skeleton Dragons...)
DECK26 = set(OWN)
SKIP = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower", "dagger-duchess-tower-bar",
        "tower-bar", "king-tower-bar", "bar", "bar-level", "clock", "emote", "text", "elixir", "selected",
        "skeleton-king-bar", "evolution-symbol", "ice-spirit-evolution-symbol", "skeleton-king-skill",
        "tesla-evolution-shock", "dirt", "axe", "bowl", "bomb"}
TOWERS = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower"}
BARS = {"tower-bar", "king-tower-bar"}
BUILDINGS = {"cannon", "cannon-cart", "tesla", "inferno-tower", "bomb-tower", "mortar", "x-bow", "goblin-hut",
             "furnace", "tombstone", "barbarian-hut", "elixir-collector", "goblin-cage", "goblin-drill"}
SPELLS = {"fireball", "the-log", "zap", "arrows", "poison", "rocket", "lightning", "earthquake", "giant-snowball",
          "tornado", "freeze", "rage", "heal", "graveyard", "clone", "barbarian-barrel", "royal-delivery"}
LAUNCHED = {"fireball", "rocket", "arrows", "giant-snowball"}  # fired from the King Tower
THROWN = {"the-log", "barbarian-barrel"}                        # roll from the drop point
PROJECTILES = {"fireball", "the-log", "rocket", "arrows", "giant-snowball", "barbarian-barrel"}
AIR = {"minion", "bat", "baby-dragon", "mega-minion", "balloon", "lava-hound", "lava-pup", "inferno-dragon",
       "electro-dragon", "skeleton-dragon", "phoenix-big", "phoenix-small", "flying-machine", "bat-evolution"}


def to_row(name: str, side: int, conf: float, vocab: dict[str, int], card_names: dict[str, str]) -> str | None:
    """The crforge row name of a KataCR unit, or None (likely false detection, or unknown)."""
    if side == 0:
        return OWN.get(name)
    if name not in DECK26 and conf < 0.72:
        return None
    if name in SPECIAL and SPECIAL[name] in vocab:
        return SPECIAL[name]
    cand = card_names.get(name)
    if cand and cand in vocab:
        return cand
    camel = "".join(w.capitalize() for w in name.replace("/", "-").split("-"))
    return camel if camel in vocab else None


def nms_rows(pred, conf_thres, iou_thres, names):
    """Raw output (1, 4 + classes + 1, N) -> list of (x1, y1, x2, y2, conf, name, side)."""
    import torch
    import torchvision
    p = pred[0].transpose(0, 1)  # (N, C)
    nc = p.shape[1] - 4
    box, cls, bel = p[:, :4], p[:, 4: 4 + nc - 1], p[:, 4 + nc - 1]
    conf, j = cls.max(1)
    keep = conf > conf_thres
    if not keep.any():
        return []
    box, conf, j, bel = box[keep], conf[keep], j[keep], bel[keep]
    xyxy = torch.cat([box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2], 1)
    i = torchvision.ops.batched_nms(xyxy, conf, j, iou_thres)
    out = []
    for k in i.tolist():
        x1, y1, x2, y2 = xyxy[k].tolist()
        out.append((x1, y1, x2, y2, float(conf[k]), names[int(j[k])], int(bel[k] > 0.5)))
    return out


class Detector:
    """The two KataCR detectors (each one knows half of the troops, so both are needed).

    Speedups that change nothing in the results: fused convolution + batch norm layers, half
    precision and cuDNN autotuning on an NVIDIA GPU, and both detectors run at the same time on the
    GPU (two CUDA streams)."""

    def __init__(self, device: str | None = None, conf: float = 0.4, iou: float = 0.6, fuse: bool = True,
                 precision: str = "auto"):
        import torch
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device, self.conf, self.iou = device, conf, iou
        self.models = []
        for w in WEIGHTS:
            ck = torch.load(str(w), map_location="cpu", weights_only=False)
            m = (ck.get("ema") or ck["model"]).float().eval()
            if fuse:
                try:
                    m = m.fuse(verbose=False)
                except TypeError:
                    m = m.fuse()
            self.models.append((m.to(device), dict(m.names)))
        self.half = False
        self.streams = None
        self.alternate = False   # one detector per frame, taking turns, if the GPU is too slow
        self.turn = 0
        self.cache = [[], []]
        self.precision = "fp32"
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            self.streams = [torch.cuda.Stream() for _ in self.models]
            # FP16 is not always faster (GTX 16xx cards have no tensor cores, drivers), so we measure
            if precision in ("auto", "fp16"):
                dummy = np.zeros((BG_H, BG_W, 3), dtype=np.uint8)
                t32 = self._bench(dummy) if precision == "auto" else 1e9
                fp32 = self.models
                self.models = [(copy_half(m), n) for m, n in fp32]
                self.half = True
                t16 = self._bench(dummy)
                if t16 <= t32:
                    self.precision = "fp16"
                    del fp32
                else:
                    self.models, self.half = fp32, False
                self.bench = {"fp32_ms": round(t32, 1) if t32 < 1e9 else None, "fp16_ms": round(t16, 1)}

    def _bench(self, img, n=3):
        import time
        self.raw(img)
        self.raw(img)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            self.raw(img)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1000

    def raw(self, arena_bgr: np.ndarray, only=None):
        """Raw detector outputs (before the confidence filter); only = the index of a single one."""
        torch = self.torch
        img = arena_bgr[:, :, ::-1].copy()
        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(self.device)
        x = (x.half() if self.half else x.float()) / 255.0
        outs = []
        models = self.models if only is None else [self.models[only]]
        streams = self.streams if only is None else None
        with torch.no_grad():
            if streams is not None:  # both detectors at the same time on the GPU
                cur = torch.cuda.current_stream()
                for (m, names), st in zip(models, streams):
                    st.wait_stream(cur)
                    with torch.cuda.stream(st):
                        pred = m(x)
                        outs.append((pred[0] if isinstance(pred, (list, tuple)) else pred, names))
                for st in self.streams:
                    cur.wait_stream(st)
            else:
                for m, names in models:
                    pred = m(x)
                    outs.append((pred[0] if isinstance(pred, (list, tuple)) else pred, names))
        return outs

    def __call__(self, arena_bgr: np.ndarray, conf: float | None = None):
        """arena_bgr: the cropped arena, 576 x 896. Returns (x1, y1, x2, y2, conf, name, side)."""
        torch = self.torch
        c = self.conf if conf is None else conf
        if self.alternate:
            # one detector on this frame, the other on the next: keep the last result of each
            i = self.turn % 2
            self.turn += 1
            (pred, names), = self.raw(arena_bgr, only=i)
            self.cache[i] = nms_rows(pred.float(), c, self.iou, names)
            rows = self.cache[0] + self.cache[1]
        else:
            rows = []
            for pred, names in self.raw(arena_bgr):
                rows += nms_rows(pred.float(), c, self.iou, names)
        if not rows:
            return []
        import torchvision
        b = torch.tensor([r[:4] for r in rows])
        s = torch.tensor([r[4] for r in rows])
        keep = torchvision.ops.nms(b, s, self.iou).tolist()
        return [rows[k] for k in keep]


def copy_half(m):
    import copy
    return copy.deepcopy(m).half()


def pixel_to_tile(px: float, py: float) -> tuple[float, float]:
    """Point of the arena image (576 x 896) -> (x, y) in tiles, in the player's frame."""
    return (px - GRID[0]) / CELL_W, 32 - (py - GRID[1]) / CELL_H


def crop_corners_tiles():
    """Corners of the arena image, in tiles in the player's frame (top left, bottom right)."""
    return pixel_to_tile(0, 0), pixel_to_tile(BG_W, BG_H)


def bar_fill(arena: np.ndarray, box, side: int, kind: str = "tower-bar") -> float | None:
    """Fill (0 to 1) of a detected health bar: red for the enemy, blue for you.

    Checked on the Nov 18 captures: enemy towers within 0.03 of their real HP."""
    import cv2
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    c = arena[max(0, y1):y2 + 1, max(0, x1):x2 + 1]
    h, w = c.shape[:2]
    if h < 3 or w < 6:
        return None
    c0 = 0.24 if kind != "bar" else 0.0  # skip the crown on the left of tower bars
    band = c[int(h * 0.2):, int(w * c0):]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV).astype(int)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue = ((H >= 165) | (H <= 8)) if side == 1 else ((H >= 92) & (H <= 118))
    full = hue & (S > 80) & (V > 165)
    empty = hue & (S > 50) & (V > 45) & (V <= 145)
    fc, ec = full.sum(0), empty.sum(0)
    state = np.where((fc >= 2) & (fc >= ec), 1, np.where(ec >= 2, 0, -1))
    idx = np.flatnonzero(state >= 0)
    if len(idx) < 4:
        return None
    s = state[idx[0]: idx[-1] + 1]
    best, bk = -1, 0
    for k in range(len(s) + 1):  # the cut that best splits "full" on the left from "empty" on the right
        score = int((s[:k] == 1).sum() + (s[k:] == 0).sum())
        if score > best:
            best, bk = score, k
    return bk / len(s)


class Tracker:
    """Tracking from one frame to the next: stable id, smoothed position and speed, number of frames
    the unit was seen in, and a VOTE over time on the class and the side.

    Position and speed: alpha-beta filter (a fixed-gain Kalman filter, constant speed). We predict
    where the unit should be, match the detection closest to that prediction (better for fast
    troops), then correct. A lost unit is kept in memory for 1.2 s."""

    ALPHA, BETA = 0.45, 0.15  # tuned on a simulated test: position within 0.2 tile, speed within 0.6 tile/s

    def __init__(self, first_id: int = 100):
        self.tracks = {}
        self.next_id = first_id

    def update(self, units, now: float):
        from collections import Counter
        used = set()
        for u in sorted(units, key=lambda u: -u["conf"]):  # most confident first
            best, bc = None, None
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                dt = min(1.2, now - tr["t"])
                px, py = tr["x"] + tr["vx"] * dt, tr["y"] + tr["vy"] * dt
                lim = 1.2 + 1.5 * dt
                d = ((px - u["x"]) ** 2 + (py - u["y"]) ** 2) ** 0.5
                if d >= lim:
                    continue
                cost = d + (0 if tr["name"] == u["name"] else 0.8) + (0 if tr["side"] == u["side"] else 1.2)
                if bc is None or cost < bc:
                    best, bc = tid, cost
            if best is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"name": u["name"], "side": u["side"], "x": u["x"], "y": u["y"], "vx": 0.0, "vy": 0.0,
                                    "t": now, "seen": 0, "born": now, "x0": u["x"], "y0": u["y"],
                                    "names": Counter(), "sides": Counter()}
                best = tid
            tr = self.tracks[best]
            used.add(best)
            tr["names"][u["name"]] += u["conf"]
            tr["sides"][u["side"]] += u["conf"]
            dt = max(1e-3, now - tr["t"])
            if tr["seen"] == 0:
                x, y, vx, vy = u["x"], u["y"], 0.0, 0.0
            else:
                px, py = tr["x"] + tr["vx"] * dt, tr["y"] + tr["vy"] * dt
                rx, ry = u["x"] - px, u["y"] - py
                x, y = px + self.ALPHA * rx, py + self.ALPHA * ry
                vx = tr["vx"] + self.BETA * rx / dt
                vy = tr["vy"] + self.BETA * ry / dt
                vmax = 4.0  # tiles per second: nothing on the arena moves faster
                vx, vy = max(-vmax, min(vmax, vx)), max(-vmax, min(vmax, vy))
            tr.update(x=x, y=y, vx=vx, vy=vy, t=now, seen=tr["seen"] + 1)
            tr["name"] = tr["names"].most_common(1)[0][0]
            tr["side"] = tr["sides"].most_common(1)[0][0]
            u["id"], u["name"], u["side"] = best, tr["name"], tr["side"]
            u["x"], u["y"] = round(x, 2), round(y, 2)
            u["dx"], u["dy"] = vx * 0.25, vy * 0.25  # move per decision (0.25 s), as in crforge
            u["seen"], u["born"], u["x0"], u["y0"] = tr["seen"], tr["born"], tr["x0"], tr["y0"]
        for tid in [t for t, tr in self.tracks.items() if now - tr["t"] > 1.2]:
            del self.tracks[tid]
        return units


class Clocks:
    """Deploy clocks (class "clock"): the game draws one at the exact point where a card was just
    played, while it deploys. The first time we see it gives the time of the play, its position
    gives the point."""

    def __init__(self):
        self.items = []  # dict(side, x, y, t, last)

    def update(self, clocks, now):
        """clocks: (side, x, y). Returns the clocks that are new in this frame."""
        new = []
        for side, x, y in clocks:
            near = [c for c in self.items if c["side"] == side and abs(c["x"] - x) < 1.5 and abs(c["y"] - y) < 1.5
                    and now - c["last"] < 1.0]
            if near:
                near[0]["last"] = now
                continue
            c = {"side": side, "x": x, "y": y, "t": now, "last": now, "used": False}
            self.items.append(c)
            new.append(c)
        self.items = [c for c in self.items if now - c["last"] < 4.0]
        return new

    def match(self, side, x, y, t, window=(-2.0, 0.6), radius=3.0):
        """The closest unused clock to a unit that appeared at (x, y) around t."""
        best = None
        for c in self.items:
            if c["used"] or c["side"] != side or not (window[0] <= c["t"] - t <= window[1]):
                continue
            d = ((c["x"] - x) ** 2 + (c["y"] - y) ** 2) ** 0.5
            if d < radius and (best is None or d < best[0]):
                best = (d, c)
        return best[1] if best else None
