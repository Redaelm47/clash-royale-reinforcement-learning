"""Shared environment for the scripts: 2.6 Hog Cycle on the RoyaleGym Rust engine.

RoyaleLearn workers import this module, so no torch at module level.
"""

import os

from royalegym import make_env
from royalegym.reward import CombinedReward, TowerHPReward, WinLossReward

# Card names in the engine (engine.cards()[i].name)
HOG_26 = ["HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons", "Fireball", "Log"]

# Weight of the tower damage bonus, next to +1 / -1 for the win.
# Read from an environment variable so the workers see the same value.
TOWER_WEIGHT_VAR = "RG26_TOWER_WEIGHT"
DEFAULT_TOWER_WEIGHT = 0.3


def build_env():
    """2.6 vs 2.6 (mirror). Replays set their own decks at reset."""
    tower = float(os.environ.get(TOWER_WEIGHT_VAR, DEFAULT_TOWER_WEIGHT))
    reward = CombinedReward([(WinLossReward(), 1.0), (TowerHPReward(), tower)])
    return make_env(reward=reward, deck=HOG_26)


def learner_26(save_dir, *, device="cpu", channels=64, blocks=4, **kwargs):
    """A Learner on this environment. The clone and PPO must use the same network."""
    from royalelearn import Learner

    return Learner(
        build_env,
        save_dir=save_dir,
        device=device,
        trunk_channels=channels,
        trunk_blocks=blocks,
        **kwargs,
    )


def find_shards(path):
    """The folder that holds manifest.json: the given path or its only subfolder."""
    from pathlib import Path

    path = Path(path)
    if (path / "manifest.json").is_file():
        return path
    found = sorted(p.parent for p in path.glob("*/manifest.json"))
    if len(found) != 1:
        raise SystemExit(f"{path}: expected one demo folder (manifest.json), found {len(found)}")
    return found[0]
