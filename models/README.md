# models/

The network code (`policy.py`). Weights are not stored here: checkpoints go in `runs/`, ignored by git.

- attention encoder over the entities;
- card head: do nothing, 4 cards, 2 hero abilities;
- tile head: 18 x 32 grid, conditioned on the chosen card;
- illegal moves are masked;
- value head, plus an optional privileged critic input (training only).

About 1.2 M parameters by default, no LSTM.

Export a checkpoint to ONNX for the crforge desktop app (`--check` compares torch and ONNX outputs):

```bash
python models/export_onnx.py --model runs/ppo/ppo_best.pt --out runs/bot_26.onnx \
    --check data/crforge_bc/rows_s00_000000.npz
```
