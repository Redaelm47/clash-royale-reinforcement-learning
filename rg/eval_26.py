"""Win rate of a bot against the clone and against scripted bots, in the 2.6 mirror.

--bot takes a PPO run folder (latest checkpoint), a specific checkpoint
(runs/ppo_26/checkpoints/000012345678) or a clone. Each opponent plays --matches
games, half on each side, with the same seeds on both sides.

--save DIR keeps one game per opponent (.msgpack); --mp4 also records it as a video.
To watch it (this needs a display, so run it on your own machine):
    royaleviser DIR/bot_vs_clone.msgpack

Example:
    python eval_26.py --bot runs/ppo_26 --clone runs/clone_26 --vs clone,random,push --matches 50
"""

import argparse
import json
import time
from pathlib import Path

from env_26 import build_env


def opponent(name, clone):
    from royalegym import NoopOpponent, RandomLegalOpponent
    from royalegym.opponents import ladder
    from royalegym.selfplay import CallableOpponent
    from royalelearn import Learner

    if name == "clone":
        if not clone:
            raise SystemExit("--vs clone needs --clone")
        bot = Learner.load_policy(clone)
        return CallableOpponent(lambda obs, mask: bot(obs)), bot
    if name == "random":
        return RandomLegalOpponent(), "random"
    if name == "noop":
        return NoopOpponent(), "noop"
    bots = dict(ladder())
    key = name.replace("_", "-")
    if key not in bots:
        raise SystemExit(f"unknown opponent {name!r}: clone, random, noop, {', '.join(bots)}")
    return bots[key], key


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bot", required=True, help="PPO run, checkpoint or clone")
    p.add_argument("--clone", help="clone folder (for --vs clone)")
    p.add_argument("--vs", default="clone,random,push", help="comma-separated opponents")
    p.add_argument("--matches", type=int, default=20, help="games per opponent (rounded to even)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", help="folder where one game per opponent is kept (.msgpack)")
    p.add_argument("--mp4", action="store_true", help="also record the kept game as a video (no display needed)")
    args = p.parse_args()

    from royalegym import evaluate, play_battle
    from royalegym.selfplay import CallableOpponent
    from royalelearn import Learner

    bot = Learner.load_policy(args.bot)
    me = CallableOpponent(lambda obs, mask: bot(obs))
    results = {}
    for name in args.vs.split(","):
        other, playable = opponent(name.strip(), args.clone)
        t0 = time.monotonic()
        res = evaluate(me, other, build_env, games=args.matches, seed=args.seed, names=("bot", name))
        took = time.monotonic() - t0
        low, high = res.interval
        print(res.summary())
        print(f"  {took / res.games:.1f} s per game", flush=True)
        results[name] = {
            "games": res.games, "wins": res.wins, "losses": res.losses, "draws": res.draws,
            "undecided": res.undecided, "win_rate": round(res.win_rate, 4),
            "interval95": [round(low, 4), round(high, 4)], "score_rate": round(res.score_rate, 4),
            "mean_crown_margin": round(res.mean_crown_margin, 3),
        }
        if args.save:
            out = Path(args.save) / f"bot_vs_{name}.msgpack"
            out.parent.mkdir(parents=True, exist_ok=True)
            env = build_env()
            battle = play_battle(env, bot, playable, seed=args.seed, save_to=out)
            env.close()
            print(f"  saved game: {battle.path} (crowns {battle.crowns}); "
                  f"to watch it: royaleviser {battle.path}")
            if args.mp4:
                from royaleviser.capture import capture
                from royaleviser.sources import open_source

                video = out.with_suffix(".mp4")
                capture(open_source(str(out)), video, scale=16, fps=20)
                print(f"  video: {video}")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
