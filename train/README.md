# train/

Training, in this order:

1. **Imitation** (`bc.py`) on the rows from `sim/il_to_crforge.py`. The converter keeps only part of the "wait" decisions (only about 4% of the steps contain a play), and the wait choice is corrected by a bias when the model is saved.
2. **PPO self-play** (`ppo.py`) against itself and a pool of older versions, starting from the imitation bot.

Reward: win or loss, plus a tower damage bonus that decays. A KL penalty toward the imitation bot decays too.

So far only the 2.6 mirror (2.6 against 2.6) is trained.

```bash
python train/bc.py --data data/crforge_bc --out runs/bc --device cuda --epochs 6
python train/ppo.py --init runs/bc/bc_best.pt --out runs/ppo --install $INSTALL --tables $TABLES \
    --envs 512 --threads 20 --device cuda --max-hours 7.5
```

Both stop cleanly with `--max-hours` and restart with `--resume`. See `python train/ppo.py --help` for the real game conditions (`--levels`, `--random-slots`, `--obs-noise`, `--action-delay`) and the opponent options (`--pfsp`, `--human-opponent`, `--moving-anchor`, `--privileged-critic`).
