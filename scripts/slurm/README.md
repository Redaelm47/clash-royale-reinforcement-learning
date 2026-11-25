# scripts/slurm/

`run_pipeline.sh` runs the training pipeline on a Slurm cluster, one stage per job:

| Stage | Job | Output |
| --- | --- | --- |
| `download` | login node (internet) | `data/replays/il_replay_all/` |
| `convert` | CPU | `data/il_all/` (all games, both players) |
| `bc-baseline` | GPU | `runs/bc_large/` (large network, 2.6 games only, control run) |
| `bc-pretrain` | GPU | `runs/bc_all/` (imitation on all games) |
| `bc-finetune` | GPU | `runs/bc_all_26/` (2.6 fine-tuning from `bc_all`) |
| `ppo A\|B\|C` | GPU + many CPU cores | `runs/ppo_A/`, ... (A: permanent tower bonus, B: + gamma 0.999, C: + the imitation bot as an opponent) |

```bash
export INSTALL=$PWD/sim/crforge/env/build/install/env TABLES=$PWD/runtime/tables/16.635.8
bash scripts/slurm/run_pipeline.sh download
sbatch --cpus-per-task=50 --time=6:00:00 --wrap "bash scripts/slurm/run_pipeline.sh convert"
sbatch --gres=gpu:1 --cpus-per-task=8 --time=4:00:00 --dependency=afterany:<convert job id> \
    --wrap "bash scripts/slurm/run_pipeline.sh bc-pretrain"
```

The header of the script shows a full `#SBATCH` job file and the variables (threads, time budgets). The simulator runs on the CPU and limits PPO speed, so PPO needs at least `THREADS` + 2 cores. `bc-baseline` and `bc-finetune` also need `data/crforge_bc/` (the 2.6 games converted with `sim/il_to_crforge.py`). `scripts/pack_for_cluster.sh` builds the archive to copy to the cluster.
