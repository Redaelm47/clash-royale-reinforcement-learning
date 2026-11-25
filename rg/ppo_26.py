"""Self-play PPO (RoyaleLearn) on the 2.6 mirror, starting from the clone.

- start: the clone weights (warm_start), with the actor frozen for --freeze-steps steps
  while the critic learns;
- stay close to the clone early on: KL to the clone (reference_kl, budget --kl-budget)
  and, if --demos is given, an imitation loss on the demos (demo_bc, --bc-coef). Both
  are relaxed over --anchor-steps steps (KL budget -> --kl-budget-end, demo_bc -> 0);
- opponents: 50% mirror (the bot against itself), 35% pool of frozen versions,
  15% scripted bots. A frozen version joins the pool every --snapshot-every steps
  (no qualification match: the RoyaleLearn "gate" costs hours);
- games run in --workers processes (--envs-per-worker games each);
- checkpoint every --checkpoint-every steps, and on stop (Ctrl-C, SIGTERM, --max-hours).
  Running the same command again with --resume continues from the last checkpoint.

Example (one GPU node, 20 cores):
    python ppo_26.py --clone runs/clone_26 --demos runs/demos/all --out runs/ppo_26 --resume \\
        --device cuda --workers 18 --envs-per-worker 32 --steps-per-update 65536 \\
        --ram-mb 30000 --steps 300000000 --max-hours 7.5

--steps counts learned decisions; --freeze-steps, --anchor-steps, --snapshot-every and
--checkpoint-every count game steps (1 game step = about 1.5 decisions).
"""

import argparse
import hashlib
import os
import signal
import threading
import time
from pathlib import Path

from env_26 import DEFAULT_TOWER_WEIGHT, TOWER_WEIGHT_VAR, find_shards, learner_26

NEVER = 10**15


def clone_net(clone):
    """Width and depth of the clone network (policy.json), to build the same network."""
    import json

    net = json.loads((Path(clone) / "policy.json").read_text())["net"]
    return int(net["channels"]), int(net["blocks"])


def extensions(args):
    """The warm_start and imitation sections for RoyaleImitate."""
    from royaleimitate.artifacts import artifact_digest

    if not args.clone:
        return {}
    clone = str(Path(args.clone).resolve())
    digest = artifact_digest(clone)
    # tolerance for the clone self-test: the PPO GPU does not give bit-exact results like the clone GPU
    warm = {"init": {"path": clone, "sha256": digest, "self_test_atol": args.self_test_atol}}
    if args.freeze_steps > 0:
        warm["actor_lr_scale"] = {"kind": "piecewise", "points": [[0, 0.0], [args.freeze_steps, 1.0]]}
    out = {"warm_start": warm}

    def fade(start, end):
        """Constant, or from start to end over --anchor-steps steps (game steps, not rows)."""
        if args.anchor_steps <= 0:
            return {"kind": "constant", "value": start}
        return {"kind": "linear", "start": start, "end": end, "over_env_steps": args.anchor_steps}

    regs = []
    if args.kl_budget > 0:
        regs.append({"kind": "reference_kl", "name": "kl_clone", "reference": "clone",
                     "budget": fade(args.kl_budget, args.kl_budget_end)})
    if args.demos and args.bc_coef > 0:
        shards = find_shards(args.demos).resolve()
        sha = hashlib.sha256((shards / "manifest.json").read_bytes()).hexdigest()
        regs.append({"kind": "demo_bc", "name": "bc_demos",
                     "shards": [{"path": str(shards), "sha256": sha}],
                     "coef": fade(args.bc_coef, 0.0),
                     "batch_rows": args.bc_rows})
    if regs:
        out["imitation"] = {
            "references": {"clone": {"kind": "snapshot", "path": clone, "sha256": digest,
                                     "self_test_atol": args.self_test_atol}},
            "regularisers": regs,
        }
    return out


def stop_later(run_dir, hours):
    """Writes 'q' to the control file: RoyaleLearn finishes the update, saves and stops."""
    def stop(*_):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "control").write_text("q")
        print(f"stop requested ({time.strftime('%H:%M:%S')}): checkpoint then exit", flush=True)

    signal.signal(signal.SIGTERM, stop)
    if hours:
        timer = threading.Timer(hours * 3600, stop)
        timer.daemon = True
        timer.start()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="run folder (checkpoints, metrics.jsonl)")
    p.add_argument("--clone", help="clone folder (clone_26.py); without it, start from scratch")
    p.add_argument("--demos", help="demos for demo_bc (optional, with --clone)")
    p.add_argument("--resume", action="store_true", help="continue the run already in --out")
    p.add_argument("--allow-drift", action="store_true",
                   help="resume even if the GPU (A100 / H100...) or torch changed since the last job")
    p.add_argument("--steps", type=int, required=True, help="target total of learned decisions (all jobs)")
    p.add_argument("--max-hours", type=float, default=0, help="clean stop after N hours")
    p.add_argument("--device", default="auto", help="cuda, cpu or auto")
    p.add_argument("--threads", type=int, default=None, help="torch threads in the main process")
    # parallel games
    p.add_argument("--workers", type=int, default=4, help="simulation processes (0 = all in one)")
    p.add_argument("--envs-per-worker", type=int, default=16, help="games per worker (even)")
    p.add_argument("--ram-mb", type=int, default=6500, help="RAM budget checked at startup")
    p.add_argument("--overlap", action="store_true",
                   help="collect the next iteration during the update (faster, 2x the buffer memory)")
    # network (taken from the clone with --clone)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    # PPO
    p.add_argument("--steps-per-update", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=None, help="default: steps-per-update / 2")
    p.add_argument("--minibatch-size", type=int, default=None, help="default: steps-per-update / 8")
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ent-coef", type=float, default=None, help="RoyaleLearn default: 0.01 -> 0.003")
    p.add_argument("--tower-weight", type=float, default=None,
                   help=f"weight of the tower damage bonus (default {DEFAULT_TOWER_WEIGHT})")
    # stay close to the clone
    p.add_argument("--freeze-steps", type=int, default=0, help="actor frozen for N game steps")
    p.add_argument("--self-test-atol", type=float, default=1e-3,
                   help="allowed gap between the reloaded clone and the saved clone")
    p.add_argument("--kl-budget", type=float, default=0.1, help="max KL to the clone (0 = off)")
    p.add_argument("--kl-budget-end", type=float, default=1.0, help="max KL at the end of --anchor-steps")
    p.add_argument("--bc-coef", type=float, default=0.1, help="demo_bc weight (0 = off)")
    p.add_argument("--bc-rows", type=int, default=64, help="demo rows per minibatch")
    p.add_argument("--anchor-steps", type=int, default=20_000_000,
                   help="game steps over which the pull toward the clone is relaxed (0 = constant)")
    # pool and checkpoints
    p.add_argument("--snapshot-every", type=int, default=2_000_000, help="game steps between two frozen versions")
    p.add_argument("--scripted", default="random_legal,push,patient", help="scripted bots")
    p.add_argument("--mix", default="0.5,0.35,0.15", help="mirror, pool, scripted shares")
    p.add_argument("--checkpoint-every", type=int, default=1_000_000, help="game steps between two checkpoints")
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    if args.tower_weight is not None:
        os.environ[TOWER_WEIGHT_VAR] = str(args.tower_weight)  # the workers read it too
    import royalelearn.config as cfg

    out = Path(args.out)
    channels, blocks = clone_net(args.clone) if args.clone else (args.channels, args.blocks)
    workers = max(0, args.workers)
    games = args.envs_per_worker
    learner = learner_26(
        out, device=args.device, channels=channels, blocks=blocks, threads=args.threads,
        n_envs=games, resume=args.resume, opponent="self", seed=args.seed,
        checkpoint_every=args.checkpoint_every, n_checkpoints_to_keep=args.keep_checkpoints,
        steps_per_update=args.steps_per_update, ppo_epochs=args.ppo_epochs,
        ppo_batch_size=args.batch_size, ppo_minibatch_size=args.minibatch_size,
        policy_lr=args.lr, critic_lr=args.lr, ppo_ent_coef=args.ent_coef,
        log_to_wandb=args.wandb, wandb_project_name="royalegym-26",
        extensions=extensions(args),
        _coordinator_kwargs={"allow_identity_drift": True} if args.allow_drift else None,
    )
    # The Learner plays everything in one process: replace that with the worker farm.
    if workers:
        learner._settings["rollout"] = cfg.RolloutConfig(
            source="process", workers=workers, games_per_worker=games,
            shards_per_worker=2, eval_workers=1, overlap=args.overlap,
        )
    else:
        learner._settings["rollout"] = cfg.RolloutConfig(
            source="inline", workers=1, games_per_worker=games, shards_per_worker=1, eval_workers=1,
        )
    mix = tuple(float(x) for x in args.mix.split(","))
    learner._settings["ladder"] = cfg.LadderConfig(
        mix=mix,
        scripted_opponents=tuple(f"scripted:{s}" for s in args.scripted.split(",")),
        candidate_every_env_steps=NEVER,  # no gate
        floor_admit_every_env_steps=args.snapshot_every,  # one frozen version every N steps
        pfsp_source="games",
    )
    learner._settings["doctor"] = cfg.DoctorConfig(ram_budget_mb=args.ram_mb)

    print(f"{max(workers, 1)} worker(s) x {games} games = {max(workers, 1) * games} games at once, "
          f"network {channels} channels x {blocks} blocks", flush=True)
    stop_later(out, args.max_hours)
    started = time.monotonic()
    learner.learn(total_steps=args.steps)
    took = time.monotonic() - started
    print(f"done: {learner.steps} steps in total, {took / 60:.1f} min in this job"
          + (f", resumed from {learner.resumed_from}" if learner.resumed_from else ""))


if __name__ == "__main__":
    main()
