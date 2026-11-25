# eval/

- `play.py`: a checkpoint against another checkpoint, the crforge scripted bot or a random bot. Prints wins, losses, draws and mean game length.
- `league.py`: Elo ranking of several checkpoints, with each version's worst score against older ones (to see if the bot forgets).
- `head_to_head.py`: two checkpoints face to face, with perfect input or `--real-game` (PC kit noise and 0.25 s delay). Also prints cards played per minute. It uses `sim/crforge/env/build/install/env` and `~/.crforge/tables/16.635.8`.

```bash
python eval/play.py --model runs/ppo/ppo_best.pt --opponent scripted --games 200 --install $INSTALL --tables $TABLES
python eval/league.py --players runs/bc/bc_best.pt "runs/ppo/ppo_it*.pt" scripted --games 64 --install $INSTALL --tables $TABLES
python eval/head_to_head.py runs/ppo/ppo_best.pt runs/bc/bc_best.pt --games 400 --real-game --temp-a 0.5 --temp-b 0.5
```

The simulator throughput (decisions per second) is printed by `scripts/check_setup.py` and logged by `train/ppo.py` (`sps`).
