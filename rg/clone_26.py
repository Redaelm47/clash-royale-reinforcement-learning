"""Behavior cloning on the 2.6 demos written by replay_26.py.

Calls royaleimitate.cloning.clone, with two extra settings:
- --play-weight: weight of the rows where the player plays a card (action resampling);
- --cache-memory-mb / --cache-dir: where to keep the packed rows between epochs
  (in memory if they fit, otherwise on disk, about 6.5 GB for 1,000 matches with one seat).

Load the --out folder with Learner.load_policy(out) (without greedy=True: the clone rarely
plays, and in greedy mode it never plays a card). It is the starting point for ppo_26.py.

Example:
    python clone_26.py --demos runs/demos/all --out runs/clone_26 --device cuda --epochs 20
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from env_26 import find_shards, learner_26


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", required=True, help="demo folder (output of replay_26.py or merge)")
    p.add_argument("--out", required=True, help="clone folder (must not exist)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3, help="stop after N epochs without progress")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto", help="cuda, cpu or auto")
    p.add_argument("--threads", type=int, default=None, help="torch threads on CPU")
    p.add_argument("--channels", type=int, default=64, help="network width (same value for PPO)")
    p.add_argument("--blocks", type=int, default=4, help="network depth (same value for PPO)")
    p.add_argument("--play-weight", type=float, default=1.0,
                   help="multiplies the weight of the rows where a card is played")
    p.add_argument("--cache", default="auto", choices=["auto", "memory", "disk", "none"])
    p.add_argument("--cache-memory-mb", type=int, default=2048,
                   help="max size of the in-memory cache before switching to disk")
    p.add_argument("--cache-dir", default=None, help="disk cache folder (default: inside --demos)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import royaleimitate.cloning as cloning
    from royaleimitate.shards import ShardReader

    demos = find_shards(args.demos)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists")

    class Reader(ShardReader):
        """ShardReader with our cache settings and the play weight."""

        def __init__(self, directory, context, *, cache="auto", **_):
            super().__init__(directory, context, cache=cache,
                             cache_memory_mb=args.cache_memory_mb, cache_dir=args.cache_dir)

        def batches(self, codec, **kw):
            if args.play_weight != 1.0:
                w = np.float32(args.play_weight)
                kw.setdefault("weight_of", lambda cols: np.where(cols["action"] != 0, w, np.float32(1)))
            return super().batches(codec, **kw)

    cloning.ShardReader = Reader

    learner = learner_26(out.parent / f".{out.name}_tmp", device=args.device,
                         channels=args.channels, blocks=args.blocks, threads=args.threads)
    started = time.monotonic()
    digest = cloning.clone(
        learner, demos, out,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        patience=args.patience, seed=args.seed,
        cache=False if args.cache == "none" else args.cache,
    )
    took = time.monotonic() - started
    spec = json.loads((out / "spec.json").read_text())
    print(json.dumps({
        "clone": str(out),
        "digest": digest,
        "seconds": round(took, 1),
        "validation_nll": spec.get("meta", {}).get("clone", {}).get("validation_nll"),
    }, indent=1))


if __name__ == "__main__":
    main()
