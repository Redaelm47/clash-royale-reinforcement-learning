"""The match timer ("Time left: 2:36", top right) read from the screen.

It gives the exact match time, even when F8 was pressed late. The game shows the time left
rounded up to the next second, and double elixir starts when it shows 1:00.
The digit templates (model/timer_digits.npz) come from the Nov 18 captures.
Checked on the 3 matches of that log: no reading error during the battle.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_DIGITS = None


def _templates():
    global _DIGITS
    if _DIGITS is None:
        f = Path(__file__).resolve().parent / "model" / "timer_digits.npz"
        _DIGITS = {int(k): v.astype(np.float32) for k, v in np.load(f).items()}
    return _DIGITS


def _digit_masks(img):
    """The 3 digits (minutes, tens, units) as 12 x 18 masks, or None."""
    H, W = img.shape[:2]
    roi = img[0:int(H * 0.10), int(W * 0.70):W]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(np.int16) for i in range(3))
    red = ((h <= 8) | (h >= 172)) & (s >= 120) & (v >= 150)  # the last 30 seconds are shown in red
    white = (v >= 190) & (s <= 70)
    n, lab, st, _ = cv2.connectedComponentsWithStats((red | white).astype(np.uint8), 8)
    comps = []
    for k in range(1, n):
        x, y, w, hh, a = st[k]
        if 0.013 * H <= hh <= 0.035 * H and 0.2 * hh <= w <= 1.1 * hh and a >= 0.15 * w * hh:
            comps.append((x, y, w, hh, k))
    best = None
    for c0 in comps:  # the row: same line, same height
        row = [c for c in comps if abs((c[1] + c[3] / 2) - (c0[1] + c0[3] / 2)) < 0.4 * c0[3]
               and abs(c[3] - c0[3]) < 0.25 * c0[3]]
        if len(row) >= 3 and (best is None or len(row) > len(best)):
            best = row
    if best is None:
        return None
    out = []
    for x, y, w, hh, k in sorted(best, key=lambda c: c[0])[-3:]:
        patch = (lab[y:y + hh, x:x + w] == k).astype(np.float32)
        side = max(w, int(hh * 0.75))  # keep the aspect ratio so a 1 stays thin
        canvas = np.zeros((hh, side), np.float32)
        canvas[:, (side - w) // 2:(side - w) // 2 + w] = patch
        out.append(cv2.resize(canvas, (12, 18), interpolation=cv2.INTER_AREA))
    return out


def read_timer(img):
    """Seconds shown by the timer (156 for "2:36"), or None if unreadable."""
    masks = _digit_masks(img)
    if masks is None:
        return None
    tpl = _templates()
    vals = []
    for p in masks:
        pv = p.ravel() - p.mean()
        scores = []
        for d, t in tpl.items():
            tv = t.ravel() - t.mean()
            den = float(np.linalg.norm(pv) * np.linalg.norm(tv))
            scores.append((float(pv @ tv) / den if den > 0 else -1.0, d))
        scores.sort(reverse=True)
        if scores[0][0] < 0.6 or scores[0][0] - scores[1][0] < 0.05:
            return None
        vals.append(scores[0][1])
    m, s1, s2 = vals
    if s1 > 5 or m > 5:
        return None
    return m * 60 + s1 * 10 + s2
