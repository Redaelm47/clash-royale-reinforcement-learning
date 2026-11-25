#!/usr/bin/env bash
# Play against the bot in the crforge desktop app.
#   1. exports the checkpoint to ONNX
#   2. opens the app: the bot plays red (top), you play blue (bottom), level 16
#
# Usage:
#   scripts/play_desktop.sh runs/play/bc_best.pt          # the imitation bot, temperature 1
#   scripts/play_desktop.sh runs/ppo/ppo_best.pt 0.7      # lower = safer card and tile choices (the pace of play does not change)
# Your deck: the 2.6 by default (mirror). For another deck, 8 card names from the tables:
#   MY_DECK=Knight,Archer,... MY_SLOTS=0,0,0,0,0,0,0,0 scripts/play_desktop.sh runs/play/bc_best.pt
#
# In the app: click a card (or keys 1 to 4), then click on the arena.
# R = new game, SPACE = pause, J = bot probabilities, K = tiles it aims at.
set -euo pipefail
MODEL="${1:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "usage: scripts/play_desktop.sh CHECKPOINT.pt [TEMPERATURE]"
  echo "checkpoints in runs/play/:"
  ls "$(dirname "$0")"/../runs/play/*.pt 2>/dev/null | xargs -n1 basename | sed 's/^/  /' || true
  exit 1
fi
MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"
cd "$(dirname "$0")/.."
ROOT="$PWD"
TEMP="${2:-1.0}"
MY_DECK="${MY_DECK:-HogRider,Musketeer,Cannon,IceGolemite,IceSpirits,Skeletons,Fireball,Log}"
MY_SLOTS="${MY_SLOTS:-0,2,1,0,0,1,0,0}"

mkdir -p runs/play
echo "ONNX export..."
.venv/bin/python models/export_onnx.py --model "$MODEL" --out runs/play/bot.onnx 2>&1 | grep -E "wrote|Error|error" || true
[ -f runs/play/bot.onnx ] || { echo "export failed"; exit 1; }

echo "opening the crforge app (bot: $MODEL, temperature $TEMP)..."
cd sim/crforge
./gradlew :desktop:run -q -Pcrforge.assetSource=cdn \
  --args="--model $ROOT/runs/play/bot.onnx --model-temperature $TEMP --model-temperature-mode inner --my-deck $MY_DECK --my-slots $MY_SLOTS"
