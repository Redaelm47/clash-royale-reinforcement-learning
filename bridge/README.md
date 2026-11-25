# bridge: from crforge to the real game

The bot learned in the crforge simulator. The bridge lets the same model play the real Clash Royale
on Google Play Games for PC (Windows). **Friendly matches only, never ladder or tournaments.**

- `pc/`: the Windows kit. User guide: [pc/README.md](pc/README.md).
- `bench/bridge_bench.py`: measures in crforge what the imperfect PC input costs the bot.

## Why Google Play Games for PC

The first plan was an Android emulator on the Mac, driven over ADB. Clash Royale installs from the
Play Store but closes right at launch: the crash log (`adb logcat -b crash`) shows an exception
`foremk.aJ: 02` thrown while a native library loads. This is the game's own protection detecting
the emulator. BlueStacks Air (5.21.790, Android 13) gives the same exception with game version
160635013. We did not try to get around this protection. ADB capture was too slow for the 2.6 deck
anyway: 0.6 to 0.8 s per frame as PNG, 0.3 to 0.6 s raw. So the kit targets Google Play Games for PC,
where the game runs officially, and works on window captures and mouse drags.

## Pipeline

One decision every 0.25 s (`pc/bot_pc.py`):

1. **Capture.** `mss` grabs the game window. A one-time calibration (two towers, the 4 card slots,
   the next card, both ends of the elixir bar) gives a linear tile to pixel map. At the start of each
   match the grid is aligned on the detected tower boxes (median offset over the first 8 frames with
   at least 3 towers).
2. **Detection** (`pc/vision.py`). The arena is cropped like KataCR's 576 x 896 images and both KataCR
   YOLOv8 detectors run on it (each knows half of the classes). Speedups with identical detections:
   fused conv + batch norm, `cudnn.benchmark`, two CUDA streams, FP16 or FP32 picked by timing both,
   and the two detectors take turns when both together need more than 150 ms. Your side is limited
   to your deck classes. On the enemy side, a class outside the 2.6 needs 0.72 confidence and 3 frames.
3. **Tracking.** An alpha-beta filter (gains 0.45 and 0.15) predicts each unit before matching it,
   votes on class and side over the track, and keeps lost units for 1.2 s. On a simulated test (0.25
   tile noise, 20 % missed frames) the position error went from 0.31 to 0.19 tile and the speed error
   from 0.90 to 0.55 tile/s, with no duplicate tracks. The spare time between decisions runs extra
   vision frames (8 to 15 per second with a GPU).
4. **Elixir, hand, timer.** Elixir: pink segments counted along the bar, plus a regen model; for 1.6 s
   after a play the bar is only trusted when it says less than expected. Hand: `cv2.matchTemplate`
   against card images learned once (0 confident errors in cross-validation between two logged
   matches), with hand and cycle tracked by logic and unrecognized cards found by elimination. Timer
   (`pc/game_timer.py`): the three digits at the top right, matched against templates, set the match
   clock. A refused play is detected when the card comes back to its slot, or when the bar has not
   dropped 2.3 s later. In a simulated match the elixir was within 0.03 on average and 39 of 44
   refused plays were detected.
5. **Observation** (`pc/state.py`). The crforge observation (7,024 ints) is rebuilt field by field, in
   the format read from a real crforge match: hand, queue, pending card, reserved elixir, refill
   delay, evolution counters; enemy plays (a new unit seen on 2 frames on their side, with point and
   time from the deploy clock), seen cards and enemy elixir counted like a player does; tower HP from
   the health bars; the exact crforge placement masks, with pockets.
6. **Digital twin** (`pc/twin.py`). crforge replays the real match live: your plays exactly, enemy
   plays as the vision saw them. crforge only plays a card that is in the hand it dealt, and the deal
   depends on the seed and the deck positions, so the kit probes the positions each seed deals and
   orders each deck so that the deal explains every play seen (`deal_order`, and `explain` for unknown
   cards and missed plays). Four variants run in parallel: variant 0 as seen, the others with small
   shifts (3 or 6 ticks, 1 tile) and another RNG. Each one is scored on every frame against the screen
   (troops seen, troops it thinks alive that the screen does not see, tower HP) and the bot gets the
   best one. The screen overrides hand, elixir, clock and tower HP, units seen only on screen are
   added, and reliable troops not seen for 1 s are removed. Rebuilds run in a second vector in the
   background.
7. **Lookahead.** For the card the bot wants to play, the sampled tile and up to 6 other top tiles are
   tried in the twin 8 s ahead, and the kit keeps the tile the value head prefers. Only the tile
   changes: a version that could also choose "wait" made the bot passive (91 % of plays cancelled).
8. **Safety checks before each click.** The bottom of the battle screen is visible, the last analyzed
   frame is less than 1 s old and a tower was seen in the last 5 s, the slot is not empty, the card is
   known, the twin is up to date, and the game window is under the mouse at both ends of the drag
   (`WindowFromPoint`). Otherwise the play is refused and logged with the reason. `play` refuses to
   start without vision, twin or learned cards, or with a vision slower than 400 ms per frame.

The decision uses temperature 0.5 in "inner" mode, as in the crforge app: the probability of waiting
stays the same, only the card and tile choices get sharper. The current model is the average of 15
PPO checkpoints, trained with card levels 9 to 16, random hero and evolution forms, noisy input and
delayed actions, so the kit gives it the real levels and forms from `settings.json`. For the older
it1450 model, trained at level 16 only, the kit scales HP to level 16 (factor 1.598 from 11 to 16).

## Measurements

**Bridge bench** (`bench/bridge_bench.py`, it1450 model, 30 games each). Same model on both sides: A
gets the input "as on the PC" (vision, twin, corrections), B gets the perfect input.

| A gets | A's score against B |
|---|---|
| the perfect input (bench control) | 43 % (9 wins, 13 losses, 8 draws) |
| the kit chain, 1 twin, no noise | 35 % |
| the kit chain, 4 variants, no noise | 43 % |
| the kit chain, 4 variants, realistic noise | 30 % |
| the kit chain, 4 variants, severe noise | 20 % |

Realistic noise: the vision misses 20 % of the units on each frame, positions are off by 0.3 tile,
and 15 % of the enemy plays are missed. Severe noise: 35 %, 0.5 and 30 %.

- With 30 games the standard deviation of a score is about 9 points. The control gives 43 % instead of
  50 %, which shows how large the randomness is.
- The chain itself costs almost nothing (4 variants: 43 %, like the control). Without noise, a field
  by field check gives an input almost identical to crforge's: the probability of waiting moves by
  0.002 and the win estimate by 0.001.
- The vision noise is what costs: about 13 points with realistic noise and 23 with severe noise. This
  is why the model was then retrained on noisy, delayed input.
- An earlier 25 % came from background rebuilds in a bench that runs much faster than real time, so
  the twin fell several seconds behind. In real time the lag is about one rebuild (300 ms on the Mac).

Run it with `--games 30 --variants 4 --miss-rate 0.2 --noise 0.3 --missed-plays 0.15`; `--perfect`
runs the control and `--background` uses background rebuilds.

**Lookahead**, against the same bot without it (on by default at 8 s, `--no-lookahead` turns it off):

| Horizon | Score |
|---|---|
| 3 s | 48 % over 40 games |
| 5 s | 58 % over 120 games |
| 8 s | **71 % over 80 games** |

**Twin validation.** A crforge match (it1450 against itself) is the ground truth. The twin gets our
exact plays and noisy enemy plays, and both are compared every 5 s.
- No noise: exact reproduction over the whole match (no gap on towers, units, their HP and enemy
  elixir). A rebuild takes 10 to 60 ms.
- Enemy plays off by up to 0.3 s and 0.5 tile: units within 0.3 tile, towers within 350 HP, enemy
  elixir exact.
- Plus 15 % of enemy plays missed: the twin drifts at times (a tower ahead or behind, a few units more
  or less, enemy elixir off by up to 5). The screen corrects the towers and adds the missing units.
- With simulated vision (a level 11 match as truth, 20 % of units missed per frame, 15 % of enemy
  plays missed, half of them leaving a deploy clock), over 4 games the bot's input holds 88 to 94 %
  of the real units, with 7 to 21 % extra units and HP within 9 to 13 %. Without missed plays, the
  twin alone finds 86 % of the units.

**Latency.**
- Mac, fixed captures: about 4 ms for cards and elixir, 2 ms for observation and decision, plus
  detection (about 250 ms). A twin rebuild takes 25 to 60 ms, stepping it one decision 0.1 ms.
- Laptop GTX 1660 Ti: 600 to 670 ms per frame for detection when power-limited to 30 W, 251 ms at 80 W.
- A real match with version 3: 360 decisions, median 165 ms, but 126 above 300 ms. The worst step
  took 1.70 s, 1.47 s of it in a twin rebuild. With background rebuilds, measured in real time over
  70 s of a match: median wait 2 ms, 95 % under 29 ms, where one decision in four used to be blocked.

## The first friendly matches

**The first friendly matches were lost**: two with an early version of the kit, then two more 3-0
with version 3. A test without clicks read the elixir 115 times out of 115 and the cards 114 times
out of 115. The logs showed these problems, all fixed since:

- **Grid offset.** The grid was 1.7 tiles too high: the calibration took the click on the middle of
  the tower drawing for the tower's base. Cards aimed at row 14, in front of the bridges, landed in
  the river or on the bridge and were refused. Fix: reference points on the drawing (about 1.3 tiles
  above the base) and alignment on the detected towers. The river now sits between rows 15 and 17.
- **Elixir animation.** The read gave one elixir less than the real value, and right after a play the
  bar empties with an animation for 0.5 to 2 s, so the bot played cards it could not pay for. Fix:
  the 1.6 s rule, the regen model and refused play detection.
- **Evolved Skeletons not recognized.** With their purple frame they stayed "?" and blocked a slot
  until the end, so the bot played with 3 cards. Fix: hand tracking by logic and elimination.
- **Observation far from training.** Missing fields (`deck_slot`, `hand_form`, the queue, enemy
  plays, reserved elixir), left and right towers swapped, enemy HP at -1. The bot saw almost no enemy
  threat and never played its Cannon. Fix: `state.py` and the twin.
- **Ghost plays from empty slots.** 9 drags started from an empty slot: the real game leaves a slot
  empty about 2 s after a play, the kit assumed 1 s like crforge. Fix: empty slot detection (0 errors
  on 1,176 real cards). All 9 ghost plays of the log are blocked and the real plays still go through.
- **Bot kept playing after the end screen.** It went on for 33 to 40 s, dragged 4 cards on the results
  screen, then on another window, and saved captures of that window. Fix: the "battle screen visible"
  check (present on every battle frame of the logs, absent on every frame after the end), the window
  check, an automatic stop at the end, and captures only during the battle.
- **Wrong clock.** F8 was pressed 22.75 s, 24.55 s and 95 s after the start, and double elixir only
  corrected one match (+14 s at once). Fix: the on-screen timer, exact within 1.6 s on the 3 logged
  matches with no reading error during the battle. F8 now only arms the bot.
- **Slow PC.** A power-limited laptop GPU, and twin rebuilds that blocked the decision (p90 700 to
  870 ms, max 5.8 s). Fix: background rebuilds, FP16 or FP32 by timing, alternating detectors, a
  vision thread, and no card when the last analyzed frame is older than 1 s (`ms.vision_age` in the log).

## Not tested yet

- No real match is logged since these fixes, so they and the current model are untested on the real
  game. On the Mac they were checked on the logged captures, without the detector and with the real
  twin; the Windows clicks and `WindowFromPoint` were not.
- Enemy play tracking at full frame rate in a real match, and `--background-click`.
- There is no vision ground truth on the real game (the crforge app does not draw the game's
  visuals): the test set is the logs, read again with `python bot_pc.py replay`. The detectors were
  trained on 2024 phone visuals and do not know the Google Play Games rendering (`docs/detector.md`).
