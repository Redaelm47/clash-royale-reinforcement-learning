# sim/

Glue between IL_Replay and the **crforge** simulator (Java engine, Python env `crforge_env` from `sim/crforge/python`).

- `cards.py`: IL_Replay card keys to crforge table names, and the 2.6 deck (`DECK_26`, `SLOTS_26`).
- `il_card_names.json`: the name table, from RoyaleImitate (MIT).
- `il_to_crforge.py`: replays IL_Replay games in crforge and writes imitation rows (observation, mask, play).
- `crforge/`: the simulator, cloned and built here (not tracked by git).

Convert the 2.6 games:

```bash
python sim/il_to_crforge.py --replays data/replays/il_replay_26/hog26_replays.parquet \
    --out data/crforge_bc --install sim/crforge/env/build/install/env \
    --tables ~/.crforge/tables/16.635.8 --threads 20 --batch 256
```

For all games and both players (pretraining), add `--all-sides`. Split the work with `--shard i --num-shards n`.
