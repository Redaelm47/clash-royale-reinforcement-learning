#!/bin/bash
# Training pipeline for a Slurm cluster, one stage per job:
#   download       the 52 raw IL_Replay files (on a login node, it needs internet)
#   convert        CPU job: replays every game in crforge, both players, for pretraining
#   bc-baseline    GPU job: the large network on the 2.6 games only (control run)
#   bc-pretrain    GPU job: imitation on all games and all decks, large network
#   bc-finetune    GPU job: 2.6 fine-tuning, starting from bc-pretrain
#   ppo A|B|C      GPU job: one PPO self-play branch
#
# Run it from the repo root, with the venv active and INSTALL and TABLES set (see README.md).
# A job script looks like this (adjust the resources to your cluster):
#   #!/bin/bash
#   #SBATCH --job-name=bc-pretrain
#   #SBATCH --gres=gpu:1
#   #SBATCH --cpus-per-task=8
#   #SBATCH --time=4:00:00
#   source .venv/bin/activate
#   export INSTALL=$PWD/sim/crforge/env/build/install/env TABLES=$PWD/runtime/tables/16.635.8
#   bash scripts/slurm/run_pipeline.sh bc-pretrain
# Chain the stages with --dependency (e.g. bc-pretrain with afterany:<convert job id>).
# The simulator runs on the CPU: give convert at least NSH x TPS cores, and ppo at least THREADS + 2.
# Time budgets in hours: CONVERT_HOURS, BASELINE_HOURS, PRETRAIN_HOURS, FINETUNE_HOURS, PPO_HOURS.
# Every stage stops cleanly when its budget runs out, so keep the job --time a bit longer.
set -u
PY=${PY:-python}
THREADS=${THREADS:-52}   # simulator threads for PPO
NSH=${NSH:-8}            # conversion processes
TPS=${TPS:-6}            # threads per conversion process
CONVERT_HOURS=${CONVERT_HOURS:-5}
BASELINE_HOURS=${BASELINE_HOURS:-1.7}
PRETRAIN_HOURS=${PRETRAIN_HOURS:-3}
FINETUNE_HOURS=${FINETUNE_HOURS:-1}
PPO_HOURS=${PPO_HOURS:-4.9}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-3}   # max epochs: the time budget is what stops it, the learning rate follows the time
LARGE="--d-model 256 --heads 8 --layers 4 --trunk 512"   # 4.6 M parameters instead of 1.2 M
BC_MODEL=${BC_MODEL:-runs/bc_all_26/bc_best.pt}   # the imitation bot used by PPO
PPO_INIT=${PPO_INIT:-$BC_MODEL}                    # the original run started from an earlier PPO checkpoint instead

download() {
  $PY scripts/download_il_replay.py data/replays/il_replay_all
}

convert() {
  N=$(ls data/replays/il_replay_all/part-*.parquet 2>/dev/null | wc -l)
  [ "$N" -eq 52 ] || echo "warning: $N IL_Replay files out of 52"
  mkdir -p data/il_all logs
  echo "conversion: $NSH processes x $TPS threads, at most $CONVERT_HOURS h ($(date +%H:%M))"
  for k in $(seq 0 $((NSH - 1))); do
    $PY sim/il_to_crforge.py --replays data/replays/il_replay_all/part-*.parquet --all-sides --out data/il_all \
      --install "$INSTALL" --tables "$TABLES" --threads "$TPS" --batch 64 --keep-wait 0.05 \
      --shard "$k" --num-shards "$NSH" --max-hours "$CONVERT_HOURS" > "logs/convert_$k.log" 2>&1 &
  done
  wait
  tail -qn 1 logs/convert_*.log
  du -sh data/il_all
}

bc_baseline() {
  # same 2.6 data as the first imitation bot, only the network is larger
  $PY train/bc.py --data data/crforge_bc --out runs/bc_large $LARGE --epochs 20 --max-hours "$BASELINE_HOURS" --device cuda
}

bc_pretrain() {
  # all games, both players, all decks
  ls data/il_all/rows_*.npz > /dev/null 2>&1 || { echo "no converted games: no pretraining"; exit 1; }
  $PY train/bc.py --data data/il_all --out runs/bc_all $LARGE --epochs "$PRETRAIN_EPOCHS" --max-hours "$PRETRAIN_HOURS" --device cuda
}

bc_finetune() {
  # 2.6 fine-tuning, starting from the pretrained model
  [ -f runs/bc_all/bc_best.pt ] || { echo "runs/bc_all/bc_best.pt not found: run bc-pretrain first"; exit 1; }
  $PY train/bc.py --data data/crforge_bc --out runs/bc_all_26 --init runs/bc_all/bc_best.pt --lr 1e-4 \
    --epochs 6 --max-hours "$FINETUNE_HOURS" --device cuda
}

ppo() {  # $1 = A, B or C, the three branches share everything else
  case "$1" in
    A) EXTRA="" ;;                                           # permanent tower damage bonus
    B) EXTRA="--gamma 0.999" ;;                              # + long horizon
    C) EXTRA="--gamma 0.999 --human-opponent $BC_MODEL" ;;   # + the real imitation bot as an opponent
    *) echo "ppo branch: A, B or C"; exit 1 ;;
  esac
  echo "branch $1: $PPO_HOURS h ($(date +%H:%M))"
  $PY train/ppo.py --init "$PPO_INIT" --install "$INSTALL" --tables "$TABLES" --threads "$THREADS" \
    --levels 9,10,11,11,11,12,13,14,15,16 --random-slots --remix-every 150 --obs-noise 0.2,0.3,0.15 --action-delay 1 \
    --pfsp --privileged-critic --pool-size 20 --pool-every 100 \
    --lr 5e-5 --kl-coef 0.025 --kl-floor 0.025 --shaping 0.2 --shaping-floor 0.2 --value-warmup 30 \
    --eval-every 150 --eval-games 128 --eval-envs 64 --eval-threads 8 --save-every 10 --keep-every 150 \
    $EXTRA --out "runs/ppo_$1" --max-hours "$PPO_HOURS"
}

case "${1:-}" in
  download) download ;;
  convert) convert ;;
  bc-baseline) bc_baseline ;;
  bc-pretrain) bc_pretrain ;;
  bc-finetune) bc_finetune ;;
  ppo) ppo "${2:-}" ;;
  *) echo "usage: run_pipeline.sh download | convert | bc-baseline | bc-pretrain | bc-finetune | ppo A|B|C"; exit 1 ;;
esac
