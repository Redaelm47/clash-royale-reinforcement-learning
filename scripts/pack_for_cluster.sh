#!/usr/bin/env bash
# Builds dist/clash_bot_cluster.tar.gz: the code, the crforge engine already built (jars + 16.635.8
# tables), the filtered 2.6 games and, if it exists, data/crforge_bc (imitation data).
# On the cluster you then only need Java 21+ and Python 3.12 (see README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
TABLES="${TABLES:-$HOME/.crforge/tables/16.635.8}"
[ -d sim/crforge/env/build/install/env ] || { echo "crforge is not built (see README.md)"; exit 1; }
[ -d "$TABLES" ] || { echo "tables not found: $TABLES"; exit 1; }
rm -rf dist/stage && mkdir -p dist/stage/clash-royale-bot/runtime/tables
S=dist/stage/clash-royale-bot
rsync -a --exclude '.venv' --exclude '__pycache__' --exclude 'runs' --exclude 'cache' \
  ./*.md .gitignore requirements.txt docs sim models train eval scripts "$S/" \
  --exclude 'sim/crforge'
mkdir -p "$S/sim/crforge/env/build/install" "$S/sim/crforge/python"
rsync -a sim/crforge/env/build/install/env "$S/sim/crforge/env/build/install/"
rsync -a --exclude '.venv' --exclude '__pycache__' sim/crforge/python/ "$S/sim/crforge/python/"
cp sim/crforge/LICENSE sim/crforge/NOTICE "$S/sim/crforge/"
rsync -a "$TABLES" "$S/runtime/tables/"
mkdir -p "$S/data/replays/il_replay_26"
cp data/replays/il_replay_26/hog26_replays.parquet data/replays/il_replay_26/hog26_summary.json "$S/data/replays/il_replay_26/"
if [ -d data/crforge_bc ] && ls data/crforge_bc/rows_*.npz >/dev/null 2>&1; then
  rsync -a data/crforge_bc "$S/data/"
fi
[ -d rg ] && rsync -a --exclude '.venv' --exclude 'runs' --exclude 'cache' rg "$S/"
tar -C dist/stage -czf dist/clash_bot_cluster.tar.gz clash-royale-bot
rm -rf dist/stage
ls -lh dist/clash_bot_cluster.tar.gz
