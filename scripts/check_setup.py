"""Checks that everything is ready for training: Java, crforge, tables, PyTorch, GPU.

Usage: python scripts/check_setup.py --install $INSTALL --tables $TABLES
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    ok = True

    java = shutil.which("java") or (os.environ.get("JAVA_HOME") and str(Path(os.environ["JAVA_HOME"]) / "bin/java"))
    if not java:
        print("[ERROR] java not found: Java 21+ is needed (module load java, or JAVA_HOME)")
        ok = False
    else:
        out = subprocess.run([java, "-version"], capture_output=True, text=True).stderr.splitlines()[0]
        print("[ok] java:", out)

    import numpy as np
    import torch

    print(f"[ok] torch {torch.__version__}, cuda: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    for p in (a.install, a.tables):
        if not Path(p).expanduser().is_dir():
            print("[ERROR] folder not found:", p)
            ok = False
    if not ok:
        sys.exit(1)

    from crforge_env import LadderConfig, SideSetup, VectorBattle
    from sim.cards import DECK_26, SLOTS_26

    v = VectorBattle(16, a.threads, install=a.install, tables=str(Path(a.tables).expanduser()), config=LadderConfig())
    print(f"[ok] crforge: tables {v.data_version}, {v.vocabulary_size} names")
    setup = SideSetup(king_level=16, tower_level=16, card_levels=[16] * 8)
    v.reset(list(range(16)), DECK_26, DECK_26, SLOTS_26, SLOTS_26, setup, setup)
    rng = np.random.default_rng(0)
    t0, n = time.time(), 0
    for _ in range(500):
        for i in range(16):
            for s in range(2):
                legal = np.flatnonzero(v.masks[i, s])
                v.actions[i, s] = rng.choice(legal) if rng.random() < 0.3 else 0
        v.step()
        n += 16
    faults = v.fault_kind_counts()
    print(f"[ok] 2.6 vs 2.6: {n} decisions in {time.time() - t0:.1f}s (Python loop included), faults {faults}")
    v.close()
    print("everything is ready")


if __name__ == "__main__":
    main()
