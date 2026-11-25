# PC kit: the 2.6 bot on the real Clash Royale

**Friendly matches and training matches only. Never ladder or tournaments.** Supercell's terms of
service forbid bots, so the account that runs the kit takes the risk.

What the kit does:
- it **reads the screen**: elixir, hand, timer, troops on the arena, towers;
- it runs the simulator the bot was trained in (crforge) **in parallel**, as a digital twin of the
  match, to fill in what the screen does not show (HP of each troop, what it targets, projectiles...);
- the bot picks its play, tries a few tiles for it in the twin (lookahead), and the **mouse** drags
  the card.

Nothing in the game is modified. The kit only reads the screen and moves the mouse, like a player.

## Requirements

- A Windows PC with Clash Royale in **Google Play Games for PC**.
- **Python 3.12**.
- **Java 21** or newer, for the twin. `install.bat` installs it.
- An **NVIDIA GPU** is strongly advised. A GTX 1660 Ti or better is enough. Without one, the vision
  runs in a separate thread with some delay, and the bot refuses to play above 400 ms per frame.
- The **2.6 deck**: Hog Rider, Musketeer, Cannon, Ice Golem, Ice Spirit, Skeletons, Fireball, The Log.

### Files that are not in the git repository

These files are ignored by git. Put them at these paths inside `bridge/pc/` before running the kit:

| File | Where it comes from |
|---|---|
| `model/detector1_v0.7.13.pt`, `model/detector2_v0.7.13.pt` | the YOLOv8 unit detectors of [KataCR](https://github.com/wty-yy/KataCR) (MIT) |
| `model/tables/` | the game data tables, version 16.635.8, generated with crforge (see `sim/crforge/docs/game-tables.md`) |
| `crforge_lite/env/lib/*.jar` | the 6 jars of the crforge `env` build (`sim/crforge/env/build/install/env/lib/`) |
| `model/bot.onnx` | the trained model, from the latest release, or exported with `models/export_onnx.py` |
| `cards/` | your card templates, made on your own PC by `python bot_pc.py cards` |

`calibration.json`, `calibration_check.png` and `logs/` are also created on each PC and stay local.

## 1. Install (once)

1. Install Python 3.12 from https://www.python.org/downloads/. On the first page of the installer,
   tick **"Add python.exe to PATH"**.
2. Update the **NVIDIA driver** (GeForce Experience or nvidia.com).
3. Put the `pc` folder somewhere simple, for example `Documents\bot_cr\pc`.
4. In the `pc` folder, double-click **`install.bat`**. It installs PyTorch with CUDA (about 3 GB),
   the other dependencies and Java 21. It takes about ten minutes.
5. Close the window. Open the `pc` folder, right-click in an empty spot, then **"Open in Terminal"**.
   All the commands below are typed there.
6. Check that everything is ready:
   ```
   python bot_pc.py check
   ```
   Each line should start with `OK`. For a `!!` line, the arrow says what to do. The `GPU` line must
   name your NVIDIA card, and the vision must take less than 150 ms per frame. Calibration and cards
   show as missing until steps 3 and 4: that is expected.

Without `install.bat`, run these commands instead:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
winget install EclipseAdoptium.Temurin.21.JRE
```

## 2. Levels and forms (`settings.json`)

The twin must simulate the match at the levels of the real cards. Open `settings.json` in Notepad.

- **Tower level**: look at the HP of your Princess Towers at the start of a match. 3,052 HP means
  level 11. Put that level in `my_tower_level`.
- **Card level**: put it in `my_card_level`. For the usual opponent, use `enemy_card_level` and
  `enemy_tower_level`. Keep 11 if you don't know.
- **Evolved cards** (purple frame) or **hero cards**: list them in `my_evolutions` and `my_heroes`
  (and `enemy_evolutions`, `enemy_heroes` for the opponent), with these names: `HogRider`,
  `Musketeer`, `Cannon`, `IceGolemite`, `IceSpirits`, `Skeletons`, `Fireball`, `Log`. For example
  `"my_evolutions": ["Skeletons", "Cannon"]`.

Keep the quotes and commas as they are. The current model was trained with varied levels and forms
(see `model/bot_info.json`), so it also gets these values: set them correctly.

## 3. Calibrate (once)

1. Start Clash Royale and open a **training match**.
2. Run:
   ```
   python bot_pc.py calibrate
   ```
3. A capture of the game opens. Click the points in the order shown in the terminal:
   - the middle of **your left tower**;
   - the middle of the **opponent's right tower**;
   - the center of your **4 cards**, from left to right;
   - the **next card** (the small one, on the left);
   - the **start**, then the **end** of the elixir bar.

   `Esc` starts over. The towers don't need to be precise: the bot aligns the grid on the detected
   towers at the start of each match.
4. Open `calibration_check.png`: the green grid must cover the arena and the red circles must land on
   the 6 towers.

Calibrate again if the size of the game window changes.

## 4. Learn your cards (once)

1. Open a **training match**.
2. Run:
   ```
   python bot_pc.py cards
   ```
   then **play normally for 90 seconds**, playing every card at least once.
3. At the end, each card image is shown. Click the image window, then type its number:
   1 Hog Rider, 2 Musketeer, 3 Cannon, 4 Ice Golem, 5 Ice Spirit, 6 Skeletons (also the ones with the
   purple evolution frame), 7 Fireball, 8 The Log, 0 to skip a bad image, `Esc` to finish.
   The number keys of a French AZERTY keyboard work without Shift.

The images are saved in `cards/` (`HogRider_1.png`, ...). If some cards are still missing, run the
command again and play them: new images are added to the old ones.

## 5. Test without clicking

1. In a training match, run:
   ```
   python bot_pc.py test
   ```
2. Press **F8**: the bot is armed and starts by itself as soon as the battle is on screen. It watches
   and decides, **without clicking anything**.
3. Check in the terminal:
   - `Troop vision:` followed by the name of your NVIDIA GPU;
   - `crforge twin ready` at start-up;
   - `Grid aligned on the towers` in the first seconds;
   - the elixir and the hand shown are right;
   - the time per step stays under 300 ms;
   - `twin yes`.
4. The bot stops by itself at the end of the battle (`Battle over`). **F9** stops it earlier.

## 6. Play

1. In a **friendly match** (or a training match), run:
   ```
   python bot_pc.py play
   ```
2. Press **F8**, then start the friendly match. The bot starts **by itself** when the battle starts:
   it reads the timer at the top right, so its clock is right even if you press F8 early. It waits a
   few seconds at the start to align the grid, then plays.
3. Don't touch the mouse during the match: the bot takes it for a fraction of a second for each card,
   then puts it back. **F8** pauses, **F9** stops.
4. At the end of the battle the bot **stops by itself** and writes the log.

One match = one command: run `play` again for the next match. Never run `play` during a ladder or
tournament battle: F8 starts the bot on whatever battle is on screen.

The bot never clicks blind. It only plays a card if the bottom of the battle screen is visible, the
card is really in its slot (not an empty slot between two cards), the twin is up to date, the game
window is under the mouse at both ends of the drag, and the vision analyzed the screen less than 1 s
ago and still sees the arena. Otherwise it prints `card NOT played` with the reason. It also refuses
to start without the vision, the twin or the learned cards (`THE BOT IS NOT PLAYING`).

Before each card, the bot tries the tile it picked and its best other tiles in the twin, 8 s ahead,
and keeps the one it likes best (lookahead). In simulation this scores 71 % against the same bot
without lookahead. `--no-lookahead` turns it off. It also turns itself off for the match if it takes
more than 400 ms three times in a row.

Other options: `--temperature` (default 0.5, lower = safer card and tile choices, the play rate does
not change), `--threaded-vision` (vision in a separate thread even when it is fast), `--title` and
`--region` (see Troubleshooting).

## 7. Check the install

`python bot_pc.py check` checks everything the bot needs and says what to fix: Python and modules,
the NVIDIA GPU, the model files, the crforge jars, Java and a test rebuild of the twin, the bot model,
`settings.json`, the calibration, the learned cards, the game window and the vision speed. Run it
first whenever something looks wrong.

## 8. Logs and replay

Each test or match is saved in `logs/<date_time>/`, then zipped to `logs/<date_time>.zip`:
- `decisions.jsonl`: one line per decision (every 0.25 s) with what was read (`read`), what the bot
  believes (`state`), the units and raw detections, the bot output (`bot`), the twin status (`twin`),
  timings (`ms`, including `ms.vision_age`), alerts, and each play or refused play;
- `info.json`: mode, window, calibration, learned cards, and at the end the grid shift, the failed
  and refused plays and how the run ended;
- `captures/`: `raw_*.jpg` (2 per second and at each play, yours or the opponent's), `play_*.jpg`
  (each of your plays, with an arrow) and `view_*.jpg` (annotated, every 2 s).

Only battle frames are saved: if another window is in front of the game, nothing is recorded.

To read a log again on the full-size captures, with its calibration:
```
python bot_pc.py replay logs\2026-11-18_035844
```
It prints what the kit reads on each frame, writes annotated captures to `<folder>/replay/`, and
sums up the raw detections. `--cards <folder>` uses another set of card images. It needs the
detectors.

## Troubleshooting

- **`Window 'Clash Royale' not found`**: check the exact title of the game window, then add
  `--title "Exact title"` to the command. As a last resort, `--region left,top,width,height`.
- **The bot does not click in the game**: the game window must be in front of the others. If that is
  not enough, run the terminal as administrator.
- **`crforge twin unavailable`**: Java is not found. `java -version` must show 21 or newer. Otherwise
  run `install.bat` again and reopen the terminal. Without the twin the bot refuses to play.
- **`THE BOT IS NOT PLAYING: some inputs are missing`**: the list below it says what is missing
  (vision, twin or learned cards).
- **`!!! PROBLEM (...)` with a beep during the match**: an input just failed (screen, vision, arena,
  elixir, hand or twin) and the message says which. Press F8 to pause, fix it (often the game window
  is hidden or moved), then F8 to resume. `--- fixed` shows when it is back.
- **A card stays `?` for a long time**: run `cards` again while playing that card.
- **`card NOT played (another window covers the game)`**: put the game in front, with nothing on top.
- **`card NOT played (vision is late ...)`** shows up often: stop with F9. The PC is too slow right now
  (see the performance tip below).
- **`the timer is unreadable`**: the bot started without reading the battle time. It still plays, but
  its clock may be off. Check the log after the match.
- **Slow, or `Slow vision: it runs in parallel`**: the vision is not using the NVIDIA GPU. Run
  `python bot_pc.py check` and follow the arrow of the `GPU` line. The bot still sees the arena, with
  some delay. Above 400 ms per frame it refuses to play.
- **`Lookahead turned off for this match`**: normal on a slow PC.
- **You want to keep your mouse**: try `python bot_pc.py play --background-click`. The clicks are then
  sent to the game window without moving the mouse. This mode is not tested: if no card gets played,
  go back to the normal mode.

### Performance tip for laptops

Plug the laptop in and switch it to its performance power mode (most makers have a key or an app for
it). On battery or in a quiet mode the GPU is power-limited and the vision slows down a lot. Measured
on a laptop GTX 1660 Ti: 600 to 670 ms per frame when limited to 30 W, 251 ms at 80 W. The kit picks
the faster of FP16 and FP32 at start-up, and alternates the two detectors when both together take
more than 150 ms.

---
The unit detectors come from KataCR (https://github.com/wty-yy/KataCR, MIT license, see
`vendor/katacr/LICENSE`). The twin uses crforge (https://github.com/voonhous/crforge, Apache-2.0
license, see `crforge_lite/LICENSE`).
