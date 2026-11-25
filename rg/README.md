# rg/: imitation then PPO with RoyaleGym (2.6 deck)

This pipeline uses the RoyaleGym tools as they are: the Rust engine (RoyaleSim), RoyaleImitate to replay IL_Replay and clone the players, and RoyaleLearn for PPO. It runs next to the crforge pipeline used in the rest of the repo.

Deck: 2.6 Hog Cycle (Hog Rider, Musketeer, Cannon, Ice Golem, Ice Spirit, Skeletons, Fireball, The Log). The engine names are `HogRider, Musketeer, Cannon, IceGolemite, IceSpirits, Skeletons, Fireball, Log`. Every game is a mirror (2.6 vs 2.6).

| File | Role |
| --- | --- |
| `env_26.py` | the shared environment (deck, reward) |
| `replay_26.py` | replays the 2.6 games of IL_Replay and writes the demos, can be split over N processes; `merge` combines them |
| `clone_26.py` | behavior cloning on the demos |
| `ppo_26.py` | self-play PPO from the clone, with a pool of frozen versions, checkpoints and resume |
| `eval_26.py` | win rate against the clone and scripted bots, with saved games to watch |
| `requirements.txt` | the exact versions that were tested |

Everything was tested end to end on a Mac on 2026-11-17, at a very small scale (75 to 80 games, 1 epoch, a few thousand PPO steps, CPU). The "Mac" numbers below come from these tests. No full training run had been done at that point, so the GPU numbers are estimates.

Run the commands from `rg/` (the scripts import `env_26.py`). Outputs go to `rg/runs/` and `rg/cache/`, which git ignores.

## 1. Setup (Linux, once)

### Python 3.12

```bash
module avail python          # if there is a 3.12 module, load it
# otherwise, without root access:
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
```

### Environment

```bash
cd clash-royale-bot/rg
uv venv --python 3.12 .venv          # or: python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
```

About 1 GB on the Mac, more on Linux (torch installs its CUDA libraries there, count on 5 GB).

### NVIDIA driver

torch 2.15.0 from PyPI needs NVIDIA driver 580 or newer (CUDA 13). To check, on a GPU node:

```bash
nvidia-smi                 # at the top: "Driver Version: 5xx.xx"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` prints `False` and the driver is older than 580, install a torch build for CUDA 12 (it works with any driver 525 or newer):

```bash
pip install --force-reinstall "torch==2.15.0" --index-url https://download.pytorch.org/whl/cu128
```

If this version does not exist for `cu128` (see the list at https://download.pytorch.org/whl/torch/), take the newest version that exists for `cu126` or `cu128`. RoyaleLearn only needs torch 2.4 or newer.

### Quick test

```bash
python -c "from env_26 import build_env; e = build_env(); e.reset(seed=0); print('ok', len(e.engine.cards()), 'cards')"
```

## 2. Data

Compute nodes may not have internet access, so download IL_Replay once from the login node (812 MB, 52 files):

```bash
hf download VanguardX101/IL_Replay --repo-type dataset --include "replays/*" --local-dir $DATA/IL_Replay
```

Without `--data-dir`, `replay_26.py` downloads the chunks it needs by itself (into `~/.cache/huggingface`, or `--hf-cache`).

## 3. Replay the 2.6 games (CPU only)

Each process takes part of the 52 chunks (`--shard i --num-shards n`, or `--chunks 0-9,15`), keeps the matches where one deck is exactly the 2.6 (`-ev1` / `-hero` suffixes removed) and only records the 2.6 seat (both seats in a mirror).

```bash
# one 20-core job is enough
for i in $(seq 0 19); do
  python replay_26.py --data-dir $DATA/IL_Replay --shard $i --num-shards 20 --out runs/demos/shard$i \
    > runs/demos/shard$i.log 2>&1 &
done
wait
python replay_26.py merge --out runs/demos/all runs/demos/shard*/
```

- The `--out` folder must be empty: to rerun a shard, delete its folder.
- Each shard writes `stats.json`: matches read, 2.6 matches kept, replayed, failed and why.
- `merge` combines the shards with hard links (no copy) and prints the sha256 of the manifest.
- `--max-matches N` stops a process after N replayed matches (for a quick try).

Measured on the Mac (1 core):

| | |
| --- | --- |
| time per replayed match | 0.15 s |
| one full chunk (5,000 matches, JSON parsing included) | 24 s for chunk 5 (155 2.6 matches) |
| memory per process | 740 MB |
| 2.6 matches that replay | 74% (269 of 362, over 7 chunks) |
| decisions recorded per match | 430 to 460 (one seat), 9 to 10% of them with a card played |
| disk | 6.3 MB per 100 matches |

Failures: 13% "play refused by the engine" (the engine refuses a move, often for lack of elixir: evolutions and heroes are played in their base form and the timing drifts), 11% "card not in the catalogue" (the opponent plays a card the engine does not have), 1% "card not in hand", less than 1% "not an eight-card deck".

The filter finds exactly the 14,040 matches with a 2.6 deck (checked on all 52 chunks). The count varies a lot from one chunk to another (34 to 521), so the shards do not all take the same time.

For the 14,040 2.6 matches of IL_Replay: about 10,400 replayed matches, 4.5 M rows, 0.7 GB. Reading the 52 chunks takes 3 s per chunk and replaying about 0.15 s per match: about 30 min on one core, a few minutes with 20 processes.

## 4. Clone (1 GPU)

```bash
python clone_26.py --demos runs/demos/all --out runs/clone_26 --device cuda \
  --epochs 12 --batch-size 1024 --cache memory --cache-memory-mb 90000
```

- The clone holds out one match in twenty for validation, stops after 3 epochs without progress (`--patience`) and keeps the best epoch.
- **Cloning cannot be resumed**: everything must fit in one 8 h job. The script prints the duration of each epoch and the maximum remaining time; set `--epochs` from that.
- `--channels` / `--blocks` set the network size (64 x 4 by default, which is 0.44 M parameters for the actor). To reach the 1 to 3 M parameters planned for the project, try `--channels 128 --blocks 6` (about 2 M from a quick estimate, roughly 6 times more compute; not tested). PPO picks up the clone size by itself.
- `--play-weight 3` gives 3 times more weight to the rows where a card is played (action resampling). At 1 (the default), the clone, when sampled, plays as often as the humans.
- Load the clone with `Learner.load_policy("runs/clone_26")`, **never with `greedy=True`**: the clone waits 90% of the time, so in greedy mode it never plays a card.

The cache of packed rows is the main disk / memory cost: 14.7 KB per row, so 6.3 GB for 1,000 matches and about **66 GB for the whole 2.6 set**. Three options:

| `--cache` | effect |
| --- | --- |
| `memory` + `--cache-memory-mb 90000` | everything in RAM, the fastest (request `--mem` of at least 100 GB) |
| `disk` + `--cache-dir /scratch/...` | on a fast local disk with 70 GB free; delete it afterwards |
| `none` | nothing is kept, everything is repacked every epoch (CPU, slower) |

By default (`auto`), the cache goes to memory if it is under 2 GB, otherwise to disk next to the demos if there is room.

Measured on the Mac (CPU, 5 threads, 75 matches = 33,464 rows): 3 to 4 min per epoch (171 s when run alone, 239 s for 80 matches), 2.9 GB peak RAM, validation NLL 6.77 before and 1.01 after one epoch. On CPU the second epoch (rows cached) is not faster, because the network is the cost. The clone folder is 26 MB (25 MB of it are test rows). On CPU the whole 2.6 set would take 6 to 8 h per epoch, so cloning runs on GPU. The RoyaleGym docs give 5 to 6 min per epoch for 1,000 matches (both seats) on an RTX 4070 Ti, so expect around 15 to 30 min per epoch for the whole 2.6 set on A100 / H100, with a longer first epoch (packing on CPU). This still has to be measured on the first job.

## 5. Self-play PPO (1 GPU + 20 cores)

```bash
python ppo_26.py --clone runs/clone_26 --demos runs/demos/all --out runs/ppo_26 --resume \
  --device cuda --workers 18 --envs-per-worker 32 --steps-per-update 65536 --minibatch-size 4096 \
  --ram-mb 30000 --freeze-steps 1000000 --steps 300000000 --max-hours 7.5
```

The next job runs **exactly the same command**. With `--resume`, the script continues from the last checkpoint in `runs/ppo_26` (and starts from scratch if the folder is empty, so you can always pass it). `--steps` is the target total over all jobs.

There are two step units:
- `--steps` counts learned decisions (one per seat played by the bot; a mirror gives two);
- `--freeze-steps`, `--anchor-steps`, `--snapshot-every` and `--checkpoint-every` count game steps (one step = 0.5 s of play for all running games). With the default mix, 1 game step = 1.5 decisions, so 300 M decisions = 200 M game steps.

What the script does:

- **Start**: the clone weights. `--freeze-steps` freezes the actor while the critic learns.
- **Stay close to the clone early on**: a KL penalty to the clone (budget `--kl-budget` 0.1) and an imitation loss on the demos (`--bc-coef` 0.1). Both are relaxed over `--anchor-steps` (20 M game steps): the KL budget rises to `--kl-budget-end` (1.0) and the imitation loss drops to 0. `--kl-budget 0` or `--bc-coef 0` turn them off.
- **Opponents** (`--mix 0.5,0.35,0.15`): 50% the bot against itself, 35% a frozen version from the pool, 15% a scripted bot (`--scripted random_legal,push,patient`). The clone is the first version in the pool, then a copy of the bot joins it every `--snapshot-every` steps (2 M). The pool picks the versions that still beat the bot more often.
- **Reward**: +1 / -1 at the end, plus `--tower-weight` (0.3) times the tower damage (as a fraction of HP). To decay it, lower `--tower-weight` from one job to the next (the script reports the change and continues).
- **Games**: `--workers` simulation processes, `--envs-per-worker` games each. The network runs on the GPU in the main process (inference and learning).
- **Clean stop**: after `--max-hours`, on SIGTERM, on Ctrl-C or with `echo q > runs/ppo_26/control`, the run finishes the current update, writes a checkpoint and stops. `echo c > runs/ppo_26/control` forces a checkpoint without stopping. On the Mac it ran up to two more updates after the request, so keep a margin (7.5 h for an 8 h job, less if an update is long). The Slurm SIGTERM at the end of the time limit usually comes 30 s before the kill, which is too late to finish an update: rely on `--max-hours`.

The RoyaleLearn "gate" (2,200 evaluation games per candidate, about 5 h each) is turned off: versions join the pool without a qualification match. So there is no "champion" or Elo in the metrics; measure with `eval_26.py` instead.

### What can change between jobs

- Fine: `--steps`, `--max-hours`, `--lr`, `--ppo-epochs`, `--batch-size`, `--minibatch-size`, `--ent-coef`, `--tower-weight`, `--ram-mb`, `--checkpoint-every`.
- Refused ("holds a run started with another ... setup"): `--workers`, `--envs-per-worker`, `--steps-per-update`, `--seed`, the network, the pool, and the clone, KL and imitation loss settings.
- **The GPU type is part of the run identity** (from the RoyaleLearn code): a run started on A100 refuses to resume on H100. Either pin the GPU type in the sbatch script, or add `--allow-drift` when resuming (tested on the Mac by changing `--seed`: the run continues and logs the differences in `drift.json`). `--allow-drift` also lets through the changes from the "refused" list, so only use it when you know what changed.

### Measurements and settings

On the Mac (CPU, 64 x 4 network):

| | |
| --- | --- |
| simulator alone, scripted bots | 2,900 game steps per second per core (5,800 decisions) |
| PPO collection, 1 game | 460 steps/s |
| PPO collection, 4 games in one process | 450 steps/s |
| PPO collection, 4 workers x 2 games | 600 to 770 steps/s (workers 10% busy: CPU inference in the main process is the bottleneck) |
| total with learning on CPU | 30 to 60 steps/s (the update takes 20 ms per step) |
| one game | 400 to 500 steps |

On the cluster, inference and updates run on the GPU. The simulator ceiling is about 18 x 2,900 = 50,000 steps/s. **Measure it on the first job**, in `runs/ppo_26/metrics.jsonl` (one line per update):

- `throughput/overall_steps_per_second`: the real speed;
- `throughput/worker_busy_frac`: close to 1 means the CPUs are the limit (add `--workers` if cores are free); low means the GPU or the main process is the limit (raise `--envs-per-worker`);
- `time/collection` and `time/update`: where the time goes.

Disk (64 x 4 network): 5.9 MB per checkpoint (3 kept), 0.9 MB per pool version (all kept: 100 for 200 M game steps, 90 MB), 8.5 KB per update in `metrics.jsonl` (40 MB for 300 M decisions), 0.6 KB per finished game in `episodes.jsonl` (about 300 MB for 200 M game steps). The `runs/ppo_26/snapshots/` folder goes with `checkpoints/`: do not copy one without the other.

What to watch in `metrics.jsonl`:

- `imitation/kl_clone/kl` and `imitation/kl_clone/budget`: the distance to the clone, which should stay under the budget;
- `imitation/bc_demos/demo_nll`: the imitation loss on the demos (1.0 at the start in the test);
- `ppo/explained_variance`: whether the critic is learning (above 0.3 before the freeze ends);
- `policy/noop_rate`: the share of decisions where the bot waits (0.92 at the start, like the clone);
- `ladder/pool_size` and `ladder/role_counts/*`: the pool grows and gets used;
- the `update` lines in the output: finished battles and mean crowns.

The settings that matter:

| Setting | Default | Effect |
| --- | --- | --- |
| `--workers`, `--envs-per-worker` | 4, 16 | parallel games; fixed for the whole run |
| `--steps-per-update` | 16,384 | steps collected before each update; 65,536 on GPU |
| `--minibatch-size` | steps / 8 | GPU memory; startup refuses if it does not fit and suggests a value |
| `--lr` | 1e-4 | learning rate (actor and critic) |
| `--freeze-steps` | 0 | actor frozen at the start |
| `--kl-budget`, `--bc-coef`, `--anchor-steps` | 0.1, 0.1, 20 M | how close to the clone the bot stays, and for how long |
| `--snapshot-every` | 2 M | how often a frozen version joins the pool |
| `--tower-weight` | 0.3 | tower damage bonus |
| `--ram-mb` | 6,500 | RAM budget checked at startup; raise it on the cluster (see Known limitations) |
| `--threads` | half the cores, 8 max | torch threads in the main process; on GPU, 2 are enough and leave the cores to the workers |
| `--overlap` | off | collects the next iteration during the update (one batch behind, twice the buffer memory); works on the Mac, worth trying on GPU if `time/update` is close to `time/collection` |

## 6. Evaluate and watch

```bash
python eval_26.py --bot runs/ppo_26 --clone runs/clone_26 --vs clone,random,push --matches 100 \
  --save runs/eval/$(date +%m%d_%H%M) --mp4
```

- `--bot` takes the run folder (latest checkpoint), a specific checkpoint (`runs/ppo_26/checkpoints/000012345678`) or a clone.
- Each opponent plays `--matches` games, half on each side, with the same seeds on both sides. The script prints the score, the win rate and its 95% interval. Available opponents: `clone`, `random`, `noop`, `first-affordable`, `defend`, `push`, `patient`.
- `--save` keeps one game per opponent (`.msgpack`, 2 to 3 MB), and `--mp4` records it as a video without a display (3 min video, 5 to 9 MB).
- On the Mac: 1 to 2 s per game (12 games and 3 saves in 26 s; a video takes about ten seconds more). With 100 games per opponent, expect 2 to 4 min per opponent.

Watching a game needs a display, so copy it to your own machine:

```bash
scp cluster:clash-royale-bot/rg/runs/eval/XXXX/bot_vs_clone.msgpack .
rg/.venv/bin/royaleviser bot_vs_clone.msgpack     # space: pause, H: help
```

or open the `.mp4` directly.

## Known limitations

- Evolutions and heroes (almost every 2.6 deck in 2026 has some) are replayed and played in their base form.
- A quarter of the 2.6 games do not replay (engine refusals, missing cards).
- Cloning cannot be resumed: it must fit in one job.
- Mirror only (2.6 vs 2.6, stage 1 of the project plan).
- The RAM budget checked at startup (`--ram-mb`) defaults to 6.5 GB. RoyaleLearn estimates about 11 GB for 18 workers x 32 games and 65,536 steps per update (8 GB with 16 games per worker): request `--mem=32G` for the job and pass `--ram-mb 30000`, otherwise the run refuses to start.
