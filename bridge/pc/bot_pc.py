"""The 2.6 bot on the real Clash Royale (Google Play Games for PC, Windows). Version 3.

FRIENDLY matches only. Never ladder or tournaments.

Commands (run in this folder):
  python bot_pc.py check           checks the install (GPU, Java, files, calibration...)
  python bot_pc.py calibrate       once: click the towers, the cards and the elixir bar
  python bot_pc.py cards           once: play a training match, then name your cards
  python bot_pc.py test            the bot watches and decides, WITHOUT clicking (to check it)
  python bot_pc.py play            the bot plays. F8 = start / pause, F9 = stop
  python bot_pc.py replay FOLDER   reads a log again (captures) and redoes the reading, to fix the kit
Each test or match is saved in logs/ (decisions + captures), then zipped.

New in version 3: the observation given to the bot is the crforge one, field by field
(hand and cycle tracked, enemy plays and elixir, tower HP, pending card...), see state.py.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from state import BY_NAME, DECK, TOWERS, Encoder, Game  # noqa: E402

CALIB = HERE / "calibration.json"
TEMPLATES = HERE / "cards"
LOGS = HERE / "logs"
MODEL = HERE / "model" / "bot.onnx"
META = HERE / "model" / "metadata.json"
WINDOWS = platform.system() == "Windows"
# Never blind during a match: no card is played if the last frame analyzed by the vision is older
# than VISION_MAX_AGE s, or if no tower has been seen for TOWER_MAX_AGE s.
VISION_MAX_AGE = 1.0
TOWER_MAX_AGE = 5.0


def vision_block(vision_age, tower_age):
    """The reason not to play a card because of the vision, or None if it is up to date."""
    if not vision_age <= VISION_MAX_AGE:  # inf or nan (no frame analyzed yet): blocked
        age = "no frame analyzed" if vision_age == float("inf") else f"frame is {vision_age:.1f} s old"
        return f"vision is late ({age}): is the PC plugged in and in performance mode?"
    if not tower_age <= TOWER_MAX_AGE:
        return f"the vision no longer sees the arena (no tower for {tower_age:.0f} s)"
    return None
# The two towers clicked during calibration: the click lands in the middle of the tower DRAWING,
# about 1.3 tiles above its base (6.5 and 25.5), like the boxes of the KataCR detector.
REF_A, REF_B = (3.5, 7.8), (14.5, 26.85)  # your left tower, enemy right tower
# Centers of the tower boxes in the KataCR arena image (576 x 896), measured on its images
TOWER_BOX = {(0, "L"): (111.5, 687.5), (0, "R"): (465.0, 686.5), (0, "K"): (287.5, 772.0),
             (1, "L"): (110.5, 210.0), (1, "R"): (465.0, 210.5)}


# -- game window -------------------------------------------------------------------------------------

def dpi_aware():
    if WINDOWS:
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass


def find_window(title="Clash Royale"):
    """Rectangle (left, top, width, height) of the window's game area, in screen pixels."""
    if not WINDOWS:
        return None
    import win32gui
    found = []

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and title.lower() in win32gui.GetWindowText(hwnd).lower():
            found.append(hwnd)
    win32gui.EnumWindows(cb, None)
    if not found:
        return None
    hwnd = found[0]
    l, t, r, b = win32gui.GetClientRect(hwnd)
    x, y = win32gui.ClientToScreen(hwnd, (0, 0))
    return {"hwnd": hwnd, "left": x, "top": y, "width": r - l, "height": b - t, "title": win32gui.GetWindowText(hwnd)}


class Screen:
    def __init__(self, args):
        import mss
        self.args = args
        self.sct = mss.mss()
        self.win = None
        self.refresh()

    def refresh(self):
        if self.args.image:
            img = cv2.imread(self.args.image)
            h, w = img.shape[:2]
            self.win = {"left": 0, "top": 0, "width": w, "height": h, "title": "image"}
            self.static = img
            return
        if self.args.region:
            l, t, w, h = map(int, self.args.region.split(","))
            self.win = {"left": l, "top": t, "width": w, "height": h, "title": "region"}
            return
        self.win = find_window(self.args.title)
        if self.win is None:
            sys.exit(f"Window '{self.args.title}' not found. Start Clash Royale in Google Play Games "
                     "(or pass --region left,top,width,height).")

    def grab(self) -> np.ndarray:
        if self.args.image:
            return self.static.copy()
        w = self.win
        shot = self.sct.grab({"left": w["left"], "top": w["top"], "width": w["width"], "height": w["height"]})
        return np.ascontiguousarray(np.array(shot)[:, :, :3])


# -- calibration ------------------------------------------------------------------------------------

STEPS = [
    "center of YOUR LEFT tower (bottom left)",
    "center of the OPPONENT's RIGHT tower (top right)",
    "center of your card 1 (leftmost in your hand)",
    "center of your card 2", "center of your card 3", "center of your card 4",
    "center of the NEXT card (the small one, left of your hand)",
    "START of the elixir bar (left end)",
    "END of the elixir bar (right end)",
]


def calibrate(args):
    scr = Screen(args)
    img = scr.grab()
    h, w = img.shape[:2]
    pts = []
    view = img.copy()
    name = "Calibration: click the points in order (Esc = start over)"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < len(STEPS):
            pts.append((x, y))
            cv2.circle(view, (x, y), 6, (0, 220, 255), -1)
            cv2.putText(view, str(len(pts)), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(name, on_mouse)
    print("Start a friendly match or a training match, and pause it if you can.")
    while len(pts) < len(STEPS):
        print(f"  {len(pts) + 1}/{len(STEPS)}: click the {STEPS[len(pts)]}", end="\r")
        cv2.imshow(name, view)
        k = cv2.waitKey(50)
        if k == 27:
            pts.clear()
            view = img.copy()
    cv2.destroyAllWindows()
    rel = [(x / w, y / h) for x, y in pts]
    calib = {"window": [w, h], "points": rel, "steps": STEPS, "date": datetime.now().isoformat(timespec="seconds")}
    CALIB.write_text(json.dumps(calib, indent=1))
    g = Geometry(calib, w, h)
    check = img.copy()
    for c in range(19):
        cv2.line(check, g.tile_px(c, 0, center=False), g.tile_px(c, 32, center=False), (60, 200, 60), 1)
    for r in range(33):
        cv2.line(check, g.tile_px(0, r, center=False), g.tile_px(18, r, center=False), (60, 200, 60), 1)
    for _, _, x, y, _ in TOWERS:
        cv2.circle(check, g.tile_px(x, y, center=False), 8, (0, 0, 255), 2)
    cv2.imwrite(str(HERE / "calibration_check.png"), check)
    print(f"\nCalibration saved. Check calibration_check.png: the green grid must cover the arena "
          "and the red circles must land on the 6 towers.")


class Geometry:
    """Tile <-> capture pixel conversions, from the calibration."""

    def __init__(self, calib, w, h):
        p = [(x * w, y * h) for x, y in calib["points"]]
        (ax, ay), (bx, by) = p[0], p[1]
        self.sx = (bx - ax) / (REF_B[0] - REF_A[0])
        self.sy = (ay - by) / (REF_B[1] - REF_A[1])
        self.ox = ax - REF_A[0] * self.sx
        self.oy = ay + REF_A[1] * self.sy
        self.cards = [tuple(map(int, q)) for q in p[2:6]]
        self.next = tuple(map(int, p[6]))
        self.elixir = (p[7], p[8])
        spacing = np.mean([np.hypot(self.cards[i + 1][0] - self.cards[i][0], self.cards[i + 1][1] - self.cards[i][1])
                           for i in range(3)])
        self.card_w = int(spacing * 0.86)
        self.card_h = int(self.card_w * 1.2)
        self.refined = False
        self.offsets = []
        g = calib.get("grid_shift")
        if g:  # shift already measured (as a fraction of the window)
            self.shift(g[0] * w, g[1] * h)
            self.refined = True

    def shift(self, dx, dy):
        self.ox += dx
        self.oy += dy

    def refine(self, rows, rect, now=None):
        """Aligns the grid on the detected towers (the first 8 frames with at least 3 towers): the
        median offset between their boxes and the ones in the KataCR images, applied to the screen."""
        if self.refined:
            return None
        x0, y0, x1, y1 = rect
        fx, fy = (x1 - x0) / 576, (y1 - y0) / 896
        d = []
        for a, b, c, e, conf, name, side in rows:
            if name not in ("queen-tower", "king-tower", "cannoneer-tower", "dagger-duchess-tower") or conf < 0.6:
                continue
            cx, cy = (a + c) / 2, (b + e) / 2
            if name == "king-tower":
                if side != 0:
                    continue  # the enemy King often changes look
                key = (0, "K")
            else:
                key = (side, "L" if cx < 288 else "R")
            ex, ey = TOWER_BOX[key]
            if abs(cx - ex) < 60 and abs(cy - ey) < 90:
                d.append((cx - ex, cy - ey))
        if len(d) >= 3:
            self.offsets.append(np.median(np.array(d), axis=0))
        if len(self.offsets) >= 8:
            dx, dy = np.median(np.array(self.offsets), axis=0)
            self.shift(dx * fx, dy * fy)
            self.refined = True
            return float(dx * fx), float(dy * fy)
        return None

    def tile_px(self, x, y, center=True):
        """Pixel of a point on the arena (in tiles); with center, x and y are a tile (col, row)."""
        if center:
            x, y = x + 0.5, y + 0.5
        return int(round(self.ox + x * self.sx)), int(round(self.oy - y * self.sy))

    def crop(self, img, center, scale=1.0, grow=1.0):
        w, h = int(self.card_w * scale * grow), int(self.card_h * scale * grow)
        x0, y0 = max(0, center[0] - w // 2), max(0, center[1] - h // 2)
        return img[y0: y0 + h, x0: x0 + w]

    def arena(self, img):
        """The arena cropped like KataCR does it (576 x 896), and the rectangle used in the capture."""
        from vision import BG_H, BG_W, crop_corners_tiles
        (tx0, ty0), (tx1, ty1) = crop_corners_tiles()
        x0, y0 = self.tile_px(tx0, ty0, center=False)
        x1, y1 = self.tile_px(tx1, ty1, center=False)
        pad = cv2.copyMakeBorder(img, 400, 400, 400, 400, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        part = pad[y0 + 400: y1 + 400, x0 + 400: x1 + 400]
        return cv2.resize(part, (BG_W, BG_H), interpolation=cv2.INTER_AREA), (x0, y0, x1, y1)


def load_geometry(img, calib=None):
    if calib is None:
        if not CALIB.exists():
            sys.exit("No calibration: run  python bot_pc.py calibrate  first")
        calib = json.loads(CALIB.read_text())
    h, w = img.shape[:2]
    return Geometry(calib, w, h)


# -- screen reading ---------------------------------------------------------------------------------

def signature(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (40, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean()
    return g / (g.std() + 1e-6)


def similarity(a, b):
    return float((a * b).mean())


def elixir_bar(img, p0, p1):
    """Number of pink segments in the elixir bar (0 to 10), or None if the bar cannot be read.

    We read a band 9 pixels high along the calibrated bar, column by column (pink = full, dark
    blue = empty), starting from the right. The number and the icon on the left do not get in the
    way. Checked on the 57 play captures from Nov 18."""
    (x0, y0), (x1, y1) = p0, p1
    h, w = img.shape[:2]
    xs = np.arange(int(x0), int(x1) + 1)
    if len(xs) < 20:
        return None
    ys = (y0 + (y1 - y0) * (xs - x0) / max(1, x1 - x0)).round().astype(int)
    band = np.stack([img[np.clip(ys + dy, 0, h - 1), np.clip(xs, 0, w - 1)] for dy in range(-4, 5)], 0).astype(int)
    b, g, r = band[..., 0], band[..., 1], band[..., 2]
    pink = ((r > 110) & (b > 110) & (g < 0.65 * np.minimum(r, b)) & (r - g > 70)).mean(0)
    navy = ((b > r + 40) & (r < 90) & (g < b - 25) & (b > 60)).mean(0)
    W = len(xs)
    if ((pink >= 0.5) | (navy >= 0.5)).mean() < 0.6:
        return None  # not the bar (menu, end screen, hidden window...)
    if navy[-6:].mean() < 0.5 and pink[-6:].mean() >= 0.5:
        return 10
    k, miss = W - 1, 0
    while k >= 0:
        if navy[k] >= 0.5:
            miss = 0
        elif pink[k] >= 0.5:
            break
        else:
            miss += 1
            if miss > 3:
                break
        k -= 1
    value = 10.0 * (k + 1) / W
    return int(min(10, max(0, round(value - 0.1))))


class Cards:
    """Card recognition by image comparison (cv2.matchTemplate), with some room for movement: a
    card that is a bit shifted or raised is still recognized."""

    MAX_PER_CARD = 8

    def __init__(self, geo: Geometry, folder: Path = TEMPLATES):
        self.geo = geo
        by = {}
        for f in sorted(folder.glob("*.png")):
            name = f.stem.rsplit("_", 1)[0]
            if name.endswith("_evo"):
                name = name[:-4]
            if name in BY_NAME:
                img = cv2.imread(str(f))
                if img is not None and img.size:
                    by.setdefault(name, []).append(img)
        self.names = sorted(by)
        self.tpl = {}  # (name, scale) -> list of grayscale templates, at half size
        self.raw = {}
        for name, imgs in by.items():
            keep, sigs = [], []
            for im in imgs:  # keep the images that differ the most from each other
                s = signature(im)
                if all(similarity(s, o) < 0.92 for o in sigs):
                    keep.append(im)
                    sigs.append(s)
                if len(keep) >= self.MAX_PER_CARD:
                    break
            self.raw[name] = keep

    def _templates(self, scale):
        key = round(scale, 3)
        if key not in self.tpl:
            w, h = int(self.geo.card_w * scale), int(self.geo.card_h * scale)
            iw, ih = max(8, int(w * 0.84 / 2)), max(8, int(h * 0.84 / 2))  # the center of the card, at half size
            out = []
            for name, imgs in self.raw.items():
                for im in imgs:
                    g = cv2.cvtColor(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                    g = cv2.resize(g, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                    y0, x0 = (g.shape[0] - ih) // 2, (g.shape[1] - iw) // 2
                    out.append((name, g[y0: y0 + ih, x0: x0 + iw]))
            self.tpl[key] = out
        return self.tpl[key]

    def empty(self, img, center):
        """True if the slot is empty (plain blue background with a crown, between playing a card and
        the next one arriving). On the Nov 18 log: 0 errors on 1,176 real cards."""
        w, h = int(self.geo.card_w * 0.7), int(self.geo.card_h * 0.7)
        x0, y0 = max(0, center[0] - w // 2), max(0, center[1] - h // 2)
        c = img[y0: y0 + h, x0: x0 + w]
        if c.size == 0:
            return False
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        blue = np.mean((hsv[:, 0] > 100) & (hsv[:, 0] < 125) & (hsv[:, 1] > 120))
        return bool(blue >= 0.7 and hsv[:, 2].std() <= 25)

    def read(self, img, center, scale=1.0):
        """(name, score from 0 to 1) of the card at this spot; name is None if nothing is close enough."""
        if not self.raw:
            return None, 0.0
        region = self.geo.crop(img, center, scale, grow=1.25)
        if region.size == 0:
            return None, 0.0
        g = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
        best = {}
        for name, t in self._templates(scale):
            if t.shape[0] > g.shape[0] or t.shape[1] > g.shape[1]:
                continue
            s = float(cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED).max())
            if s > best.get(name, -1):
                best[name] = s
        if not best:
            return None, 0.0
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        name, s = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if s < 0.55 or s - second < 0.06:  # too weak, or two cards almost tied
            return None, round(s, 3)
        return name, round(s, 3)


class Reader:
    def __init__(self, geo: Geometry, detector=None, vocab=None, card_names=None, templates=TEMPLATES):
        self.geo = geo
        self.detector, self.vocab, self.card_names = detector, vocab or {}, card_names or {}
        self.cards = Cards(geo, templates)
        self.grid_shift = None
        self.last_raw = []
        self.tracker = None
        if detector is not None:
            from vision import Tracker
            self.tracker = Tracker()

    def units(self, img, now):
        from vision import AIR, BARS, BUILDINGS, PROJECTILES, SKIP, SPELLS, TOWERS as TW, bar_fill, pixel_to_tile, to_row
        arena, rect = self.geo.arena(img)
        rows = self.detector(arena, conf=0.2)
        moved = self.geo.refine([r for r in rows if r[4] >= 0.4], rect)
        if moved is not None:
            self.grid_shift = moved
            print(f"\nGrid aligned on the towers: {moved[0]:+.1f} px, {moved[1]:+.1f} px")
            arena, rect = self.geo.arena(img)
            rows = self.detector(arena, conf=0.2)
        # every detection from 0.2 up goes to the log (to tune the thresholds on real images);
        # the bot only uses the ones above the normal threshold
        self.last_raw = [[r[5], r[6], round(r[4], 3)] + [int(round(v)) for v in r[:4]] for r in rows]
        rows = [r for r in rows if r[4] >= self.detector.conf]
        units, boxes, bars, clocks = [], [], [], []
        for x1, y1, x2, y2, conf, name, side in rows:
            if (x2 - x1) * (y2 - y1) > 0.2 * arena.shape[0] * arena.shape[1]:
                continue
            if name == "clock":  # deploy clock: the exact point of a card that was just played
                if conf >= 0.6:
                    tx, ty = pixel_to_tile((x1 + x2) / 2, (y1 + y2) / 2)
                    clocks.append((int(side), round(tx, 2), round(ty, 2)))
                continue
            if name in TW:
                tx, ty = pixel_to_tile((x1 + x2) / 2, (y1 + y2) / 2)
                boxes.append((side, tx, ty))
                continue
            if name in BARS:
                tx, ty = pixel_to_tile((x1 + x2) / 2, (y1 + y2) / 2)
                bars.append((side, name, tx, ty, bar_fill(arena, (x1, y1, x2, y2), side, name)))
                continue
            if name in SKIP:
                continue
            row = to_row(name, side, conf, self.vocab, self.card_names)
            if row is None:
                continue
            tx, ty = pixel_to_tile((x1 + x2) / 2, y1 + 0.75 * (y2 - y1))
            kind = 3 if name in PROJECTILES else (4 if name in SPELLS else 1)
            units.append({"name": name, "row": row, "side": int(side), "x": round(tx, 2), "y": round(ty, 2),
                          "conf": round(conf, 2), "kind": kind, "air": name in AIR, "building": name in BUILDINGS,
                          "box": [round(v) for v in (x1, y1, x2, y2)]})
        units = self.tracker.update(units, now)
        kept = []
        for u in units:  # after the vote, the kept class and side may have changed
            name = u["name"]
            row = to_row(name, u["side"], max(u["conf"], 0.8 if u["seen"] >= 3 else u["conf"]), self.vocab, self.card_names)
            if row is None:
                continue
            u.update(row=row, kind=3 if name in PROJECTILES else (4 if name in SPELLS else 1),
                     air=name in AIR, building=name in BUILDINGS)
            kept.append(u)
        return kept, boxes, bars, rect, clocks

    def read(self, img, now, with_units=True):
        from game_timer import read_timer
        hand = [self.cards.read(img, c) for c in self.geo.cards]
        nxt = self.cards.read(img, self.geo.next, scale=0.55)
        empties = [self.cards.empty(img, c) for c in self.geo.cards]
        bar = elixir_bar(img, *self.geo.elixir)
        out = {"hand": hand, "next": nxt, "elixir": bar, "empty": empties, "timer": read_timer(img),
               # the bottom of the battle screen is there (elixir bar, or at least 2 sure slots): otherwise it
               # is the end screen, a menu or another window, and the bot does not click. On the end screen
               # the forest sometimes looks like a card (score 0.55 to 0.65); a real card is above 0.88.
               "panel": bar is not None or sum(1 for h, e in zip(hand, empties) if (h[0] and h[1] >= 0.8) or e) >= 2}
        if self.detector is not None and with_units:
            out["units"], out["tower_boxes"], out["tower_bars"], out["arena_rect"], out["clocks"] = self.units(img, now)
            out["raw"] = self.last_raw
        return out


# -- card labelling ---------------------------------------------------------------------------------

def label_cards(args):
    scr = Screen(args)
    img = scr.grab()
    geo = load_geometry(img)
    TEMPLATES.mkdir(exist_ok=True)
    print("Play a training match and play your cards normally for ~90 s: the kit captures your cards.")
    kept, sigs = [], []
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        img = scr.grab()
        for c in list(geo.cards) + [geo.next]:
            crop = geo.crop(img, c) if c != geo.next else cv2.resize(
                geo.crop(img, c, 0.55), (geo.card_w, geo.card_h), interpolation=cv2.INTER_AREA)
            s = signature(crop)
            if all(similarity(s, o) < 0.75 for o in sigs):
                kept.append(crop)
                sigs.append(s)
        print(f"  {len(kept)} different card images, {int(t_end - time.time())} s left", end="\r")
        time.sleep(0.4)
    print()
    names = "  ".join(f"{i + 1}={d[1]}" for i, d in enumerate(DECK))
    win = "Which card? 1-8, 0 = skip, Esc = done"
    count = {}
    for f in TEMPLATES.glob("*.png"):
        n = f.stem.rsplit("_", 1)[0]
        count[n] = max(count.get(n, 0), int(f.stem.rsplit("_", 1)[1]) if f.stem.rsplit("_", 1)[1].isdigit() else 0)
    # digits, numpad, and the unshifted top row of a French AZERTY keyboard
    # (& e-acute " ' ( - e-grave _ a-grave)
    keys = {ord(str(i)): i for i in range(10)}
    keys.update({ord(c): i for c, i in zip("&\"'(-_", (1, 3, 4, 5, 6, 8))})
    keys.update({233: 2, 232: 7, 224: 0})  # e-acute, e-grave, a-grave
    print(f"{len(kept)} images to name. Click the image window, then type the number (Esc to finish).")
    print("A card with the purple evolution frame (Skeletons, Cannon) gets the same name as the normal card.")
    new = {}
    for n_img, crop in enumerate(kept):
        big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(win, big)
        print(f"image {n_img + 1}/{len(kept)}: {names}")
        k = -1
        while k not in keys and k != 27:
            k = cv2.waitKeyEx(0)
            k = k & 0xFFFF if k > 0xFFFF else k
            if k > 255 and (k & 0xFF) in keys:
                k = k & 0xFF
        if k == 27:
            break
        i = keys[k]
        if i == 0 or i > 8:
            continue
        name = DECK[i - 1][0]
        count[name] = count.get(name, 0) + 1
        new[name] = new.get(name, 0) + 1
        cv2.imwrite(str(TEMPLATES / f"{name}_{count[name]}.png"), crop)
    cv2.destroyAllWindows()
    have = {f.stem.rsplit("_", 1)[0] for f in TEMPLATES.glob("*.png")}
    missing = [d[1] for d in DECK if d[0] not in have]
    print("New images:", {BY_NAME[n][1]: c for n, c in new.items()})
    if missing:
        print("Missing:", ", ".join(missing), "-> run  python bot_pc.py cards  again and play them.")


# -- the bot ----------------------------------------------------------------------------------------

class Bot:
    def __init__(self, temperature=0.5, seed=0):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
        self.enc = Encoder(json.loads(META.read_text()))
        self.T = temperature
        self.rng = np.random.default_rng(seed)

    def decide(self, obs, mask):
        first, tile, value = self.sess.run(None, {"observations": obs[None], "masks": mask[None]})
        f = first[0].astype(np.float64)
        p = np.exp(f - f.max())
        p /= p.sum()
        p_wait = p[0]
        if self.T != 1.0 and p_wait < 1.0:
            # "inner" mode, as in the crforge app: the probability of waiting does not change
            # (otherwise the bot gets passive), only the card and tile choice gets sharper
            rest = np.where(mask_first(mask, len(p)), np.log(np.maximum(p, 1e-300)) / self.T, -np.inf)
            rest[0] = -np.inf
            if np.isfinite(rest).any():
                q = np.exp(rest - rest[np.isfinite(rest)].max())
                q[~np.isfinite(rest)] = 0
                q = q / q.sum() * (1 - p_wait)
                q[0] = p_wait
                p = q / q.sum()
        choice = int(self.rng.choice(len(p), p=p))
        info = {"wait": round(float(p_wait), 3), "win": round(float(np.clip((value[0] + 1) / 2, 0, 1)), 3),
                "cards": [round(float(x), 3) for x in p[1:5]]}
        if choice == 0 or choice > 4:
            return 0, None, info
        slot = choice - 1
        t = tile[0, slot].astype(np.float64)
        if self.T != 1.0:
            t = t / self.T
        legal = mask[1 + slot * 576: 1 + (slot + 1) * 576]
        t = np.where(legal, t, -np.inf)
        q = np.exp(t - t.max())
        q /= q.sum()
        k = int(self.rng.choice(len(q), p=q))
        self.last_tiles = tile[0, slot].astype(np.float64)  # for the tile lookahead
        return 1 + slot * 576 + k, (slot, k % 18, k // 18), info

    def value(self, obs_batch):
        """The bot's "who will win" estimate for several observations."""
        n = len(obs_batch)
        masks = np.zeros((n, self.enc.actions), dtype=bool)
        masks[:, 0] = True
        _, _, v = self.sess.run(None, {"observations": np.ascontiguousarray(obs_batch, dtype=np.int32), "masks": masks})
        return np.asarray(v, dtype=np.float64).reshape(-1)

    def refine_tile(self, twin, card, slot, k, mask, horizon=160, lam=0.3, top=6):
        """Tile lookahead: in the twin, the chosen card is played on the sampled tile and on the bot's
        best other tiles; we keep the tile whose outcome (horizon ticks later) the bot likes best.
        Only the tile changes: the timing and the card stay the bot's."""
        if twin is None or not twin.ok or getattr(self, "last_tiles", None) is None:
            return k, None
        o = twin.v.observations[twin.best, 0]
        hand = [int(x) for x in o[twin.H["hand_card"]]]
        cid = self.enc.ids.get(card)
        if cid not in hand:
            return k, None
        tslot = hand.index(cid)
        legal = mask[1 + slot * 576: 1 + (slot + 1) * 576]
        tm = twin.v.masks[twin.best, 0][1 + tslot * 576: 1 + (tslot + 1) * 576]
        t = np.where(legal & tm, self.last_tiles, -np.inf)
        cands = [k] + [int(c) for c in np.argsort(-t)[:top] if np.isfinite(t[c]) and int(c) != k]
        if len(cands) < 2 or not tm[k]:
            return k, None
        acts = np.array([1 + tslot * 576 + c for c in cands], dtype=np.int32)
        obs, _ = twin.v.lookahead(np.full(len(acts), twin.best, dtype=np.int32), np.zeros(len(acts), dtype=np.int32),
                                  acts, horizon)
        val = self.value(obs.astype(np.int32))
        score = val + lam * (t[cands] - t[cands].max())
        best = cands[int(np.argmax(score))]
        return best, {"tiles": len(cands), "change": best != k}


def mask_first(mask, n):
    """Is the first choice allowed? (wait, 4 cards, 2 abilities)"""
    out = np.zeros(n, dtype=bool)
    out[0] = True
    for s in range(4):
        out[1 + s] = mask[1 + s * 576: 1 + (s + 1) * 576].any()
    return out


class Mouse:
    """Drags the card to the tile. The cursor is put back where it was right after (~0.1 s)."""

    def __init__(self, scr: Screen, dry: bool, background: bool = False):
        self.scr, self.dry, self.background = scr, dry or not WINDOWS, background

    def _game_at(self, pt):
        """True if the game window really is under this screen point (and not another window in front)."""
        import win32gui
        try:
            h = win32gui.WindowFromPoint(pt)
            for _ in range(50):
                if h == self.scr.win["hwnd"]:
                    return True
                parent = win32gui.GetParent(h)
                if not parent:
                    return False
                h = parent
        except Exception:
            return False
        return False

    def drag(self, p0, p1):
        """Plays the card. Returns None if done, else the reason it refused (nothing was clicked)."""
        if self.dry:
            return None
        w = self.scr.win
        if not (0 <= p1[0] < w["width"] and 0 <= p1[1] < w["height"]):
            return "tile outside the window"
        if self.background:
            self._post(p0, p1)
            return None
        import win32api
        import win32con
        import win32gui
        old = win32api.GetCursorPos()
        try:
            if win32gui.GetForegroundWindow() != w["hwnd"]:
                win32gui.SetForegroundWindow(w["hwnd"])
        except Exception:
            pass
        a = (w["left"] + p0[0], w["top"] + p0[1])
        b = (w["left"] + p1[0], w["top"] + p1[1])
        # the click goes wherever the mouse is, in any window, so we check it is the game at both
        # ends of the drag (on Nov 18, after the match, the bot dragged onto another window)
        if not (self._game_at(a) and self._game_at(b)):
            return "another window covers the game"
        win32api.SetCursorPos(a)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(0.015)
        for k in range(1, 6):
            win32api.SetCursorPos((int(a[0] + (b[0] - a[0]) * k / 5), int(a[1] + (b[1] - a[1]) * k / 5)))
            time.sleep(0.01)
        time.sleep(0.012)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        time.sleep(0.01)
        win32api.SetCursorPos(old)

    def _post(self, p0, p1):
        """Experimental: clicks sent to the window without moving the real mouse (may not work)."""
        import win32api
        import win32con
        import win32gui
        hwnd = self.scr.win["hwnd"]
        child = win32gui.ChildWindowFromPointEx(hwnd, p0, 0) or hwnd

        def lp(p):
            x, y = p
            if child != hwnd:
                sx, sy = win32gui.ClientToScreen(hwnd, p)
                x, y = win32gui.ScreenToClient(child, (sx, sy))
            return win32api.MAKELONG(x & 0xFFFF, y & 0xFFFF)
        win32gui.PostMessage(child, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lp(p0))
        for k in range(1, 6):
            p = (int(p0[0] + (p1[0] - p0[0]) * k / 5), int(p0[1] + (p1[1] - p0[1]) * k / 5))
            time.sleep(0.01)
            win32gui.PostMessage(child, win32con.WM_MOUSEMOVE, win32con.MK_LBUTTON, lp(p))
        time.sleep(0.012)
        win32gui.PostMessage(child, win32con.WM_LBUTTONUP, 0, lp(p1))


def alarm():
    """A beep (Windows) so that a failure never goes unnoticed."""
    if WINDOWS:
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONHAND)
        except Exception:
            pass
    else:
        print("\a", end="")


class Health:
    """Watches every input during the match and shouts RIGHT AWAY when one of them fails."""

    # a failed read on a single frame is not a failure (the "elixir bar is full" message hides the bar,
    # a card shows up in big), so we only shout after this delay
    GRACE = {"elixir": 1.5, "hand": 1.5}

    def __init__(self):
        self.active = {}
        self.first = {}

    def check(self, now, game, st, twin_info, vision_age, tower_age, panel_age=0.0):
        problems = {}
        if panel_age > 1.0:
            # no more elixir bar or cards: the end of the battle, a menu, or another window in front
            problems["screen"] = (f"the bottom of the battle screen has not been visible for {panel_age:.0f} s "
                                  "(battle over, or game window hidden): the bot stops clicking")
        elif vision_age > 2.0:
            problems["vision"] = f"the vision has not analyzed anything for {vision_age:.0f} s"
        elif tower_age > 5.0:
            problems["arena"] = (f"no tower seen for {tower_age:.0f} s: the game window is hidden, moved, "
                                 "or the calibration is wrong")
        if panel_age <= 1.0:
            if game.elixir.e is None or st.get("elixir") is None:
                problems["elixir"] = "the elixir bar is not read (calibration? hidden window?)"
            if sum(1 for h, e in zip(st["hand"], st.get("empty") or [False] * 4) if h[0] or e) == 0 \
                    and not any(game.hand.hand):
                problems["hand"] = "no card in the hand is recognized (run  python bot_pc.py cards  again)"
        for k in list(self.first):
            if k not in problems:
                del self.first[k]
        for k in problems:
            self.first.setdefault(k, now)
        problems = {k: v for k, v in problems.items() if now - self.first[k] >= self.GRACE.get(k, 0.0)}
        if game.twin is None:
            problems["twin"] = "the crforge twin stopped: the bot no longer has its full observation"
        elif twin_info is not None and not twin_info.get("ok"):
            problems["twin"] = "the crforge twin could not rebuild the match"
        new = [k for k in problems if k not in self.active]
        for k in new:
            alarm()
            print(f"\n!!! PROBLEM ({k}): {problems[k]}")
        for k in [k for k in self.active if k not in problems]:
            print(f"\n--- fixed ({k})")
        self.active = problems
        return problems


def load_detector(args):
    """The troop detectors, on the GPU if possible, and their measured speed."""
    weights = [HERE / "model" / "detector1_v0.7.13.pt", HERE / "model" / "detector2_v0.7.13.pt"]
    if not all(w.exists() for w in weights):
        raise RuntimeError("detectors missing from model/ (unzip the full kit again)")
    import torch
    from vision import Detector
    t0 = time.time()
    det = Detector(device=args.device)
    name = torch.cuda.get_device_name(0) if det.device == "cuda" else det.device
    dummy = np.zeros((896, 576, 3), dtype=np.uint8)
    for _ in range(2):  # the first pass is slower (GPU warm-up)
        det(dummy)
    t1 = time.perf_counter()
    for _ in range(3):
        det(dummy)
    det.ms = (time.perf_counter() - t1) / 3 * 1000
    prec = f", {det.precision}" if det.device == "cuda" else ""
    if det.ms > 150:
        # too slow to run both detectors on every frame: alternate them (each one every other frame)
        det.alternate = True
        t1 = time.perf_counter()
        for _ in range(4):
            det(dummy)
        slow = det.ms
        det.ms = (time.perf_counter() - t1) / 4 * 1000
        print(f"Troop vision: {name}{prec}, {slow:.0f} ms for both detectors, so they take turns: "
              f"{det.ms:.0f} ms per frame (loaded in {time.time() - t0:.1f} s).")
    else:
        print(f"Troop vision: {name}{prec}, {det.ms:.0f} ms per frame (loaded in {time.time() - t0:.1f} s).")
    if det.device != "cuda":
        print("WARNING: the vision is not running on an NVIDIA GPU, so it will be slow. Check that")
        print("  python -c \"import torch; print(torch.cuda.is_available())\"  prints True (see README.md).")
    return det


class VisionThread:
    """The vision in a separate thread, when it is too slow to keep up with the decisions (no NVIDIA
    GPU): the bot decides every 0.25 s with the last analyzed frame, instead of playing blind or
    slowing down."""

    def __init__(self, reader, win):
        import threading
        self.reader, self.win = reader, dict(win)
        self.lock = threading.Lock()
        self.latest, self.seq, self.used = None, 0, 0
        self.t_frame = 0.0
        self.stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        import mss
        sct = mss.mss()
        w = self.win
        while not self.stop:
            try:
                shot = sct.grab({"left": w["left"], "top": w["top"], "width": w["width"], "height": w["height"]})
                img = np.ascontiguousarray(np.array(shot)[:, :, :3])
                now = time.time()
                res = self.reader.units(img, now)
                with self.lock:
                    self.latest = res
                    self.t_frame = now
                    self.seq += 1
            except Exception as e:  # never kill the bot for a failed frame
                print(f"\nvision: {type(e).__name__}: {str(e)[:120]}")
                time.sleep(0.2)

    def take(self):
        """The last result if it is new, else None."""
        with self.lock:
            if self.latest is None or self.seq == self.used:
                return None
            self.used = self.seq
            return self.latest


def draw_view(img, st, game, now):
    """Annotated capture (half size): units seen, and what the bot thinks it knows."""
    shot = img.copy()
    if "arena_rect" in st:
        x0, y0, x1, y1 = st["arena_rect"]
        fx, fy = (x1 - x0) / 576, (y1 - y0) / 896
        cv2.rectangle(shot, (x0, y0), (x1, y1), (120, 120, 120), 1)
        for u in st["units"]:
            a, b, c, d = u["box"]
            col = (255, 140, 40) if u["side"] == 0 else (40, 60, 255)
            p0 = (int(x0 + a * fx), int(y0 + b * fy))
            cv2.rectangle(shot, p0, (int(x0 + c * fx), int(y0 + d * fy)), col, 2)
            cv2.putText(shot, f"{u['name']} {u['id']}", (p0[0], p0[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cards, pend = game.hand.view(now)
    lines = [f"t {game.t(now):5.1f}s  elixir {game.elixir.e if game.elixir.e is None else round(game.elixir.e, 1)}"
             f" (bar {st.get('elixir')}) x{game.elixir.mult}  enemy {round(game.enemy.elixir or 0, 1)}",
             "hand " + " ".join((BY_NAME[c][1][:8] if c else "?") + ("*" if p else "") for c, p in zip(cards, pend)),
             "queue " + " ".join(BY_NAME[c][1][:8] if c else "?" for c in game.hand.queue),
             "enemy " + " ".join(p["card"][:8] for p in game.enemy.plays[-5:]),
             "tower HP " + " ".join(f"{int(game.towers.frac(k) * 100)}" for k in range(6))]
    for i, line in enumerate(lines):
        cv2.putText(shot, line, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(shot, line, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return cv2.resize(shot, (img.shape[1] // 2, img.shape[0] // 2))


def record(t_s, st, game, now, info, act, ms):
    cards, pend = game.hand.view(now)
    return {"t": round(t_s, 2),
            "read": {"elixir_bar": st.get("elixir"), "hand": st["hand"], "next": st["next"],
                     "empty": st.get("empty"), "timer": st.get("timer"), "panel": st.get("panel")},
            "screen_clock": game.clock_synced,
            "state": {"elixir": None if game.elixir.e is None else round(game.elixir.e, 2), "x": game.elixir.mult,
                      "hand": cards, "pending": pend, "queue": game.hand.queue,
                      "evo": game.hand.evo, "enemy_elixir": None if game.enemy.elixir is None else round(game.enemy.elixir, 2),
                      "enemy_plays": game.enemy.count, "enemy_seen": game.enemy.seen,
                      "towers": [round(game.towers.frac(k), 2) for k in range(6)], "alive": game.towers.alive,
                      "failed": game.hand.failed},
            "units": [{k: u[k] for k in ("name", "side", "x", "y", "conf", "id", "seen")} for u in st.get("units", [])],
            "raw_detections": st.get("raw", []),
            "clocks": st.get("clocks", []),
            "bot": info, "action": act, "ms": ms}


def run(args, dry: bool):
    dpi_aware()
    scr = Screen(args)
    img = scr.grab()
    geo = load_geometry(img)
    bot = Bot(args.temperature)
    try:
        import twin as _j
        if _j.VARIED_FORMS:  # model trained with varied forms: give it the real ones
            from state import configure_forms
            cfg = _j.settings()
            configure_forms(cfg["my_evolutions"], cfg["my_heroes"])
            print(f"Model with varied forms: evolutions {cfg['my_evolutions']}, heroes {cfg['my_heroes']}.")
        if _j.VARIED_LEVELS:
            print("Model with varied levels: it gets the real levels from settings.json.")
    except Exception:
        pass
    blockers = []
    try:
        detector = load_detector(args)
    except Exception as e:
        detector = None
        blockers.append(f"VISION: cannot load it ({type(e).__name__}: {str(e)[:200]})")
    if detector is not None and detector.ms > 400 and not args.image:
        blockers.append(f"VISION: {detector.ms:.0f} ms per frame, far too slow. PyTorch is not using the NVIDIA "
                        "GPU: run  python bot_pc.py check  and follow the \"GPU\" line.")
    card_names = json.loads((HERE / "model" / "il_card_names.json").read_text())
    reader = Reader(geo, detector, bot.enc.ids, card_names)
    missing = [d[1] for d in DECK if d[0] not in reader.cards.raw]
    if missing:
        blockers.append(f"CARDS: not learned: {', '.join(missing)} (run  python bot_pc.py cards)")
    else:
        print("Learned cards:", ", ".join(f"{BY_NAME[n][1]} ({len(v)})" for n, v in reader.cards.raw.items()))
    mouse = Mouse(scr, dry, args.background_click)
    twin = None
    try:
        from twin import Twin
        t0 = time.time()
        twin = Twin()
        print(f"crforge twin ready in {time.time() - t0:.1f} s ({twin.n} variants): the bot gets its full observation.")
    except Exception as e:
        blockers.append(f"TWIN: unavailable ({type(e).__name__}: {str(e)[:200]}). Install Java 21 "
                        "(install.bat), reopen the terminal, then  python bot_pc.py check")
    if blockers:
        alarm()
        print("\n" + "!" * 78)
        print("THE BOT IS NOT PLAYING: some inputs are missing. It never plays blind.")
        for b in blockers:
            print("  - " + b)
        print("!" * 78)
        sys.exit(1)
    out = LOGS / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    (out / "captures").mkdir(parents=True, exist_ok=True)
    (out / "info.json").write_text(json.dumps({
        "version": 3, "mode": "test" if dry else "play", "window": {k: v for k, v in scr.win.items() if k != "hwnd"},
        "calibration": json.loads(CALIB.read_text()), "learned_cards": {n: len(v) for n, v in reader.cards.raw.items()},
        "temperature": args.temperature, "python": sys.version, "platform": platform.platform()}, indent=1))
    log = open(out / "decisions.jsonl", "w")
    # F8 before the battle "arms" the bot: it starts by itself when the battle starts (timer at 3:00),
    # so at the right time even if F8 is pressed early. Never without F8: you pick the battle (friendly).
    state = {"on": args.image is not None or args.autostart, "quit": False, "auto": False, "started": False}
    if WINDOWS and not args.image:
        from pynput import keyboard

        def on_key(key):
            if key == keyboard.Key.f8:
                if not state["started"] and not state["on"]:
                    state["auto"] = not state["auto"]
                    print("\nbot:", "ARMED, it starts as soon as the battle starts" if state["auto"] else "disarmed")
                    return
                state["on"] = not state["on"]
                print("\nbot:", "RUNNING" if state["on"] else "paused")
            elif key == keyboard.Key.f9:
                state["quit"] = True
        keyboard.Listener(on_press=on_key, daemon=True).start()
    print(f"{'TEST (no clicks)' if dry else 'PLAY'}. F8: arm the bot (it starts by itself when the battle starts), "
          f"then pause. F9: stop. Log: {out}")
    game = None
    vthread = None
    health = Health()
    last_vision, last_tower = time.time(), time.time()
    last_enemy_count = 0
    slow_lookahead = 0
    if detector is not None and not args.image and (detector.ms > 150 or args.threaded_vision):
        vthread = VisionThread(reader, scr.win)
        print(f"Slow vision ({detector.ms:.0f} ms): it runs in parallel, the bot decides with the last frame seen.")
    last_play, last_view, last_raw, n = 0.0, 0.0, 0.0, 0
    det_ms, extra = 0.1, 0  # time of one detection (s), extra vision frames
    period = 0.25
    last_panel, panel_seen, armed_panel = None, False, None
    refused, last_refuse = 0, (None, 0.0)
    end_msg = None
    try:
        while not state["quit"]:
            t_loop = time.perf_counter()
            if not state["on"]:
                if state["auto"]:  # armed: watch for the battle (the bottom of the screen and the timer)
                    st0 = reader.read(scr.grab(), time.time(), with_units=False)
                    r = st0.get("timer")
                    if st0["panel"] and r is not None:
                        state["on"], state["auto"] = True, False
                        if r >= 170:
                            print(f"\nBattle detected ({r // 60}:{r % 60:02d}): the bot starts.")
                        else:
                            print(f"\nBattle started {180 - r} s ago: the bot starts without having seen the beginning.")
                        continue
                    # timer unreadable (window of another size?): start on the battle anyway
                    armed_panel = (armed_panel or time.time()) if st0["panel"] else None
                    if armed_panel and time.time() - armed_panel > 2.0:
                        state["on"], state["auto"] = True, False
                        alarm()
                        print("\n!!! Battle seen, but the timer is unreadable: the bot starts, its clock "
                              "may be off. Check the log after the match.")
                        continue
                    time.sleep(0.2)
                else:
                    time.sleep(0.05)
                continue
            state["started"] = True
            now = time.time()
            if game is None:
                game = Game(now, twin)
            img = scr.grab()
            t_cap = time.perf_counter()
            if vthread is not None:  # slow vision: take the last frame analyzed by the separate thread
                st = reader.read(img, now, with_units=False)
                res = vthread.take()
                if res is not None:
                    st["units"], st["tower_boxes"], st["tower_bars"], st["arena_rect"], st["clocks"] = res
                    st["raw"] = list(reader.last_raw)
                st["image_age_ms"] = round((now - vthread.t_frame) * 1000) if vthread.t_frame else None
            else:
                st = reader.read(img, now)
            t_read = time.perf_counter()
            game.update(st, now)
            game.geo_ready = geo.refined or detector is None or game.t(now) > 6.0
            obs = game.observation(bot.enc, now)
            twin_info = None
            if game.twin is not None:
                t_tw = time.perf_counter()
                try:
                    from twin import add_unseen, drop_ghosts, view
                    enc = bot.enc
                    if game.twin.sync(now, game.t0):
                        game.twin.judge(game.units, [game.towers.frac(k) for k in range(6)], now)
                        full = view(game.twin, game.twin.observation(), obs, enc)
                        full, ghosts = drop_ghosts(full, game.twin, game.units, now, enc)
                        obs, added = add_unseen(full, obs, enc.H, enc.E, enc.header_ints, enc.entity_ints)
                        twin_info = {"ok": True, "added": added, "ghosts": ghosts, "variant": game.twin.best,
                                     "scores": [round(x, 2) for x in game.twin.score]}
                    else:
                        twin_info = {"ok": False}
                    twin_info.update(rebuilds=game.twin.rebuilds, dropped=game.twin.dropped,
                                     ms=round((time.perf_counter() - t_tw) * 1000, 1))
                except Exception as e:  # the twin crashed: carry on without it
                    print(f"\nTwin stopped ({type(e).__name__}: {str(e)[:200]})")
                    twin_info = {"ok": False, "error": str(e)[:300]}
                    game.twin = None
            mask = game.mask(bot.enc.actions, now)
            act, play, info = bot.decide(obs, mask)
            if play and not args.no_lookahead and game.twin is not None:
                try:  # lookahead: the best tile for the chosen card, tried in the twin
                    slot, col, row = play
                    cards_now, _ = game.hand.view(now)
                    t_r = time.perf_counter()
                    k2, rinfo = bot.refine_tile(game.twin, cards_now[slot], slot, row * 18 + col, mask)
                    if rinfo:
                        rinfo["ms"] = round((time.perf_counter() - t_r) * 1000, 1)
                        slow_lookahead = slow_lookahead + 1 if rinfo["ms"] > 400 else 0
                        if slow_lookahead >= 3:  # the PC is too slow: a late play costs more than the lookahead brings
                            args.no_lookahead = True
                            print("\n--- Lookahead turned off for this match: over 400 ms three times in a row.")
                        play, act = (slot, k2 % 18, k2 // 18), 1 + slot * 576 + k2
                        info["lookahead"] = rinfo
                except Exception as e:
                    info["lookahead"] = {"error": str(e)[:200]}
            t_dec = time.perf_counter()
            t_s = game.t(now)
            ms = {"capture": round((t_cap - t_loop) * 1000, 1), "read": round((t_read - t_cap) * 1000, 1),
                  "decision": round((t_dec - t_read) * 1000, 1)}
            rec = record(t_s, st, game, now, info, act, ms)
            rec["twin"] = twin_info
            rec["ms"]["image_age"] = st.get("image_age_ms", rec["ms"]["read"])
            if game.twin is not None and game.twin.build_ms:
                rec["ms"]["twin_rebuild"] = game.twin.build_ms[-1]
            if "units" in st:
                last_vision = now
                if st.get("tower_boxes"):
                    last_tower = now
            if st.get("panel"):
                last_panel, panel_seen = now, True
            elif last_panel is None:
                last_panel = now
            panel_age = now - last_panel
            if vthread is not None:  # the age of the last frame really analyzed by the vision thread
                vision_age = now - vthread.t_frame if vthread.t_frame else float("inf")
            else:
                vision_age = now - last_vision
            rec["ms"]["vision_age"] = None if vision_age == float("inf") else round(vision_age * 1000)
            if game.t(now) > 3.0 or panel_age > 1.0:
                rec["alerts"] = health.check(now, game, st, twin_info, now - last_vision, now - last_tower, panel_age)
            if play and now - last_play >= 0.25:
                slot, col, row = play
                cards, _ = game.hand.view(now)
                card = cards[slot]
                p0, p1 = geo.cards[slot], geo.tile_px(col, row)
                # never blind: the battle on screen, the card in its slot, the twin up to date,
                # and the game window under the mouse (checked by mouse.drag)
                if not st.get("panel"):
                    why = "the bottom of the battle screen is not visible"
                elif vision_block(vision_age, now - last_tower):
                    why = vision_block(vision_age, now - last_tower)
                elif (st.get("empty") or [False] * 4)[slot]:
                    why = "this slot is empty on screen"
                elif card is None:
                    why = "unknown card"
                elif not (twin_info and twin_info.get("ok")):
                    why = "the twin is not up to date"
                else:
                    why = mouse.drag(p0, p1)
                if why:
                    rec["refused_play"] = {"card": card, "slot": slot, "reason": why}
                    refused += 1
                    if why != last_refuse[0] or now - last_refuse[1] > 3.0:
                        print(f"\n{t_s:6.1f} s  card NOT played ({why})", end="")
                        last_refuse = (why, now)
                    play = None
            if play and now - last_play >= 0.25:
                game.on_play(slot, card, (col, row), now)
                rec["play"] = {"card": card, "slot": slot, "tile": [col, row], "from": p0, "to": p1}
                last_play = now
                dbg = img.copy()
                cv2.circle(dbg, p0, 10, (0, 220, 255), 2)
                cv2.arrowedLine(dbg, p0, p1, (0, 220, 255), 2)
                cv2.imwrite(str(out / "captures" / f"play_{n:04d}.jpg"), dbg, [cv2.IMWRITE_JPEG_QUALITY, 80])
                print(f"\n{t_s:6.1f} s  plays {BY_NAME[card][1]} on tile ({col}, {row})", end="")
            # raw capture (2 per second, and on every play, yours or the opponent's): enough to replay
            # and tune the whole reading later on real images
            burst = "play" in rec or game.enemy.count != last_enemy_count
            last_enemy_count = game.enemy.count
            # battle frames only: if another window is in front, nothing is saved
            if st.get("panel") and (now - last_raw >= 0.5 or burst):
                cv2.imwrite(str(out / "captures" / f"raw_{n:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                last_raw = now
            if st.get("panel") and now - last_view >= 2.0:
                cv2.imwrite(str(out / "captures" / f"view_{n:04d}.jpg"), draw_view(img, st, game, now),
                            [cv2.IMWRITE_JPEG_QUALITY, 70])
                last_view = now
            rec["ms"]["total"] = round((time.perf_counter() - t_loop) * 1000, 1)
            rec["extra_vision_frames"] = extra
            log.write(json.dumps(rec) + "\n")
            n += 1
            cards, pend = game.hand.view(now)
            nb = [sum(1 for u in st.get("units", []) if u["side"] == s) for s in (0, 1)]
            e = game.elixir.e
            print(f"\r{t_s:6.1f} s  elixir {'?' if e is None else int(e)}  hand "
                  f"{[BY_NAME[c][1][:10] if c else '?' for c in cards]}  units {nb[0]}/{nb[1]}  enemy {game.enemy.count} plays"
                  f"  win {int(info['win'] * 100)} %  {int(rec['ms']['total'])} ms"
                  f"  twin {'yes' if twin_info and twin_info.get('ok') else 'no'}   ", end="")
            # end of the battle: the bottom of the screen is gone but the arena is still there (end-of-battle
            # banner), or nothing from the battle for 20 s. The bot stops by itself and writes the log.
            if panel_seen and panel_age >= 2.0 and now - last_tower < 1.5 and now - last_vision < 1.5:
                end_msg = "Battle over: the bot stops by itself."
                break
            if panel_seen and panel_age >= 20.0:
                end_msg = "The battle has not been on screen for 20 s: the bot stops."
                break
            if args.image and n >= args.steps:
                break
            # the time left before the next decision is used to look at the arena more often
            # (8 to 15 frames per second with the GPU): more chances to see each unit
            while detector is not None and vthread is None and not args.image:
                left = period - (time.perf_counter() - t_loop)
                if left < det_ms * 1.3:
                    break
                t_v = time.perf_counter()
                img2 = scr.grab()
                now2 = time.time()
                u2, b2, bars2, _, c2 = reader.units(img2, now2)
                game.update_vision(u2, b2, bars2, c2, now2)
                extra += 1
                det_ms = 0.8 * det_ms + 0.2 * (time.perf_counter() - t_v)
            time.sleep(max(0.0, period - (time.perf_counter() - t_loop)))
    except KeyboardInterrupt:
        pass
    if end_msg:
        print("\n\n--- " + end_msg)
    if vthread is not None:
        vthread.stop = True
    log.close()
    meta = json.loads((out / "info.json").read_text())
    meta["grid_shift_px"] = reader.grid_shift
    meta["failed_plays"] = game.hand.failed if game else None
    meta["refused_plays"] = refused
    meta["end"] = end_msg or "F9"
    meta["screen_clock"] = bool(game and game.clock_synced)
    if twin is not None:
        meta["twin"] = {"rebuilds": twin.rebuilds, "dropped": twin.dropped, "errors": twin.errors[-10:]}
    (out / "info.json").write_text(json.dumps(meta, indent=1))
    z = out.with_suffix(".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out.rglob("*"):
            zf.write(f, f.relative_to(out.parent))
    print(f"\nDone. Log zipped to: {z}")


def replay(args):
    """Redoes the reading on the full-size captures of a log (play_* and raw_*), in order, with
    that log's calibration. Writes the annotated captures to FOLDER/replay/."""
    folder = Path(args.folder)
    info = json.loads((folder / "info.json").read_text())
    recs = [json.loads(line) for line in open(folder / "decisions.jsonl")]
    shots = sorted([f for f in (folder / "captures").glob("*.jpg") if f.stem.split("_")[0] in ("play", "raw")],
                   key=lambda f: int(f.stem.split("_")[1]))
    if not shots:
        sys.exit("No full-size capture (play_* or raw_*) in this log.")
    first = cv2.imread(str(shots[0]))
    geo = Geometry(info["calibration"], first.shape[1], first.shape[0])
    enc = Encoder(json.loads(META.read_text()))
    detector = load_detector(args)
    card_names = json.loads((HERE / "model" / "il_card_names.json").read_text())
    tpl = Path(args.cards) if args.cards else TEMPLATES
    reader = Reader(geo, detector, enc.ids, card_names, tpl)
    dest = folder / "replay"
    dest.mkdir(exist_ok=True)
    game, base = None, time.time()
    for f in shots:
        k = int(f.stem.split("_")[1])
        if k >= len(recs):
            continue
        now = base + recs[k]["t"]
        if game is None:
            game = Game(now)
        img = cv2.imread(str(f))
        st = reader.read(img, now)
        game.update(st, now)
        obs = game.observation(enc, now)
        play = recs[k].get("play")
        if play and play["card"] in BY_NAME:
            slot = play.get("slot", geo.cards.index(tuple(play["from"])) if tuple(play["from"]) in geo.cards else 0)
            game.on_play(slot, play["card"], tuple(play["tile"]), now)
        cards, pend = game.hand.view(now)
        nb = [sum(1 for u in st.get("units", []) if u["side"] == s) for s in (0, 1)]
        print(f"{f.stem}  t {recs[k]['t']:6.1f}  bar {st['elixir']}  elixir {None if game.elixir.e is None else round(game.elixir.e, 1)}"
              f"  read {[(h[0] or '?')[:6] + f'{h[1]:.2f}' for h in st['hand']]}  hand {[(c or '?')[:6] for c in cards]}"
              f"  queue {[(c or '?')[:6] for c in game.hand.queue]}  units {nb}  enemy {[p['card'][:6] for p in game.enemy.plays[-4:]]}"
              f"  towers {[int(game.towers.frac(k) * 100) for k in range(6)]}  entities {int(obs[enc.H['entity_count'][0]])}")
        cv2.imwrite(str(dest / f"{f.stem}.jpg"), draw_view(img, st, game, now), [cv2.IMWRITE_JPEG_QUALITY, 75])
    # vision summary for this log (to tune the thresholds on real images)
    from collections import Counter
    classes, sides, low = Counter(), Counter(), Counter()
    for r in recs:
        for name, side, conf, *_ in r.get("raw_detections", []):
            classes[name] += 1
            sides[(name, side)] += 1
            if conf < 0.4:
                low[name] += 1
    if classes:
        print("\nRaw detections in the log (class: total, of which below 0.4):")
        for name, c in classes.most_common(25):
            print(f"  {name:24s} {c:6d}  {low[name]:6d}")
    if game is not None:
        print(f"\nEnemy plays spotted: {game.enemy.count} ({game.enemy.from_clock} from a deploy clock), "
              f"cards seen: {game.enemy.seen}, your plays refused: {game.hand.failed}")
    print(f"Annotated captures: {dest}")


def check(args):
    """Checks everything the bot needs and says what to fix."""
    problems = []

    def good(msg):
        print(f"  OK   {msg}")

    def bad(msg, fix):
        print(f"  !!   {msg}\n         -> {fix}")
        problems.append(msg)
    print("Kit check\n")
    v = sys.version_info
    (good if v[:2] >= (3, 10) else lambda m: bad(m, "install Python 3.12"))(f"Python {v.major}.{v.minor}.{v.micro}")
    mods = ["numpy", "cv2", "mss", "onnxruntime", "ultralytics", "torch", "torchvision", "jpype"]
    if WINDOWS:
        mods += ["win32gui", "pynput"]
    for m in mods:
        try:
            mod = __import__(m)
            good(f"module {m} {getattr(mod, '__version__', '')}")
        except Exception as e:
            fix = "install Java, then: pip install -r requirements.txt" if m == "jpype" else "pip install -r requirements.txt"
            if m in ("torch", "torchvision"):
                fix = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
            bad(f"module {m} missing ({type(e).__name__})", fix)
    if np.__version__.startswith("2"):
        bad(f"numpy {np.__version__} (numpy 1.x is needed)", 'pip install "numpy<2"')
    try:
        import torch
        if torch.cuda.is_available():
            good(f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        else:
            bad(f"PyTorch {torch.__version__} does not see an NVIDIA GPU: the vision will be slow",
                "pip uninstall -y torch torchvision, then pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu124 (and an up-to-date NVIDIA driver)")
    except Exception:
        pass
    for f in ["model/bot.onnx", "model/metadata.json", "model/detector1_v0.7.13.pt", "model/detector2_v0.7.13.pt",
              "model/cards_info.json", "model/masks.json", "model/twin_template.json", "model/tables/globals.json"]:
        (good if (HERE / f).exists() else lambda m: bad(m, "unzip the full kit again"))(f"file {f}")
    jars = list((HERE / "crforge_lite" / "env" / "lib").glob("*.jar"))
    (good if len(jars) >= 6 else lambda m: bad(m, "unzip the full kit again"))(f"crforge: {len(jars)} jar files")
    try:
        sys.path.insert(0, str(HERE))
        from crforge_lite._jvm import find_jvm
        good(f"Java: {find_jvm()}")
        from twin import Twin
        t0 = time.time()
        tw = Twin()
        t1 = time.time()
        tw.add_own("HogRider", 1000 + 6.0, (3, 14))
        tw.add_enemy("Musketeer", 1000 + 8.0, 9.5, 20.5)
        ok = tw.sync(1000 + 12.0, 1000.0)
        good(f"crforge twin: {tw.n} variants ready in {t1 - t0:.1f} s, rebuild in "
             f"{(time.time() - t1) * 1000:.0f} ms" + ("" if ok else " (rebuild FAILED)"))
        if not ok:
            problems.append("twin")
    except Exception as e:
        bad(f"crforge twin unavailable ({type(e).__name__}: {str(e)[:120]})",
            "winget install EclipseAdoptium.Temurin.21.JRE, then reopen the terminal")
    try:
        import onnxruntime as ort
        ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
        good("bot model (bot.onnx) loaded")
    except Exception as e:
        bad(f"bot model unreadable ({type(e).__name__})", "unzip the full kit again")
    try:
        r = json.loads((HERE / "settings.json").read_text())
        good(f"settings.json: card level {r['my_card_level']}, tower level {r['my_tower_level']}, "
             f"evolutions {r['my_evolutions']}, heroes {r['my_heroes']}")
    except Exception as e:
        bad(f"settings.json unreadable ({type(e).__name__})", "fix the file (quotes, commas)")
    if CALIB.exists():
        good("calibration done (calibration.json)")
    else:
        bad("no calibration", "python bot_pc.py calibrate (in a training match)")
    have = {}
    for f in TEMPLATES.glob("*.png"):
        n = f.stem.rsplit("_", 1)[0]
        have[n] = have.get(n, 0) + 1
    missing = [d[1] for d in DECK if d[0] not in have]
    if TEMPLATES.exists() and not missing:
        good("learned cards: " + ", ".join(f"{BY_NAME[n][1]} {c}" for n, c in have.items() if n in BY_NAME))
    else:
        bad("cards to learn: " + (", ".join(missing) if missing else "all"),
            "python bot_pc.py cards (in a training match)")
    win = find_window(args.title) if WINDOWS else None
    if WINDOWS:
        if win:
            good(f"game window found: '{win['title']}' {win['width']} x {win['height']}")
        else:
            bad(f"window '{args.title}' not found", "start Clash Royale, or add --title \"Exact title\"")
    det = None
    try:
        det = load_detector(args)
        if det is not None:
            (good if det.ms <= 150 else lambda m: bad(m, "slow vision: check the NVIDIA GPU (see above)"))(
                f"vision: {det.ms:.0f} ms per frame" + (" (fast enough)" if det.ms <= 150 else ""))
    except Exception as e:
        bad(f"vision cannot be loaded ({type(e).__name__}: {str(e)[:120]})", "pip install -r requirements.txt")
    if win and det is not None and CALIB.exists():
        try:
            scr = Screen(args)
            img = scr.grab()
            geo = load_geometry(img)
            reader = Reader(geo, det, Encoder(json.loads(META.read_text())).ids,
                            json.loads((HERE / "model" / "il_card_names.json").read_text()))
            st = reader.read(img, time.time())
            print(f"         game capture: elixir read {st['elixir']}, hand {[h[0] for h in st['hand']]}, "
                  f"{len(st['units'])} units, {len(st['tower_boxes'])} towers seen")
        except Exception as e:
            print(f"         (cannot read a capture: {type(e).__name__}: {str(e)[:100]})")
    print()
    if problems:
        print(f"{len(problems)} problem(s) to fix before playing (see the arrows).")
    else:
        print("All set. Run  python bot_pc.py test  in a training match.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "calibrate", "cards", "test", "play", "replay"])
    ap.add_argument("folder", nargs="?", help="for replay: the folder of a log")
    ap.add_argument("--title", default="Clash Royale", help="title of the game window")
    ap.add_argument("--region", default=None, help="left,top,width,height if the window is not found")
    ap.add_argument("--image", default=None, help="test on a screenshot (for development)")
    ap.add_argument("--steps", type=int, default=20, help="with --image: number of decisions")
    ap.add_argument("--seconds", type=int, default=90, help="capture time for \"cards\"")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="lower = safer card and tile choices; the play rate does not change (default 0.5)")
    ap.add_argument("--autostart", action="store_true", help="start without waiting for F8")
    ap.add_argument("--device", default=None, help="cuda, cpu or mps (default: the fastest available)")
    ap.add_argument("--background-click", action="store_true",
                    help="experimental: clicks sent to the window without taking the mouse (may not work)")
    ap.add_argument("--cards", default=None, help="for replay: another folder of card images")
    ap.add_argument("--threaded-vision", action="store_true", help="run the vision in a separate thread even if it is fast")
    ap.add_argument("--no-lookahead", action="store_true",
                    help="do not try tiles in the twin before playing a card (lookahead: +21 points "
                         "in simulation, see bridge/README.md)")
    args = ap.parse_args()
    dpi_aware()
    if args.command == "check":
        check(args)
    elif args.command == "calibrate":
        calibrate(args)
    elif args.command == "cards":
        label_cards(args)
    elif args.command == "replay":
        if not args.folder:
            sys.exit("Give the log folder:  python bot_pc.py replay logs\\2026-11-18_035844")
        replay(args)
    else:
        run(args, dry=args.command == "test")


if __name__ == "__main__":
    main()
