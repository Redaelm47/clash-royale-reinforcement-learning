"""PPO self-play in crforge, starting from the imitation bot.

- --envs games played at once (VectorBattle on --threads threads), 2.6 against 2.6.
- A share of the games (--self-frac): the bot plays both seats, and both are used for learning.
- The others: the bot against an older version drawn from the pool (at first the imitation
  bot, then a copy of the bot every --pool-every iterations).
- Reward: +1 win, -1 loss at the end, plus a tower damage bonus (damage dealt minus damage
  taken) that decays to 0 over --shaping-iters iterations.
- KL penalty toward the imitation bot, which also decays (--kl-iters), so the bot does not
  forget everything early on. With --moving-anchor, this reference model (the anchor) moves
  forward when the bot clearly beats it in the evaluation.
- Saves every --save-every iterations; --resume restarts from ppo_last.pt; --max-hours
  stops cleanly before the job's time limit.

Usage:
  python train/ppo.py --init runs/bc/bc_best.pt --out runs/ppo --install sim/crforge/env/build/install/env \
      --tables ~/.crforge/tables/16.635.8 --envs 512 --threads 20 --device cuda --max-hours 7.5
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.play import make_opponent, model_policy, run_games  # noqa: E402
from models.policy import autocast, ENT, HEADER, Layout, entropy, kl, load, log_prob, sample, save  # noqa: E402
from sim.cards import DECK_26, SLOTS_26  # noqa: E402

from crforge_env import LadderConfig, SideSetup, VectorBattle  # noqa: E402


def linear(it, start, iters):
    return 0.0 if iters <= 0 else start * max(0.0, 1.0 - it / iters)


class Opponents:
    """The pool of older versions and the loaded opponent models (one per group of games)."""

    def __init__(self, base_model, groups: int, size: int, device, pfsp: bool = False):
        self.size, self.device = size, device
        self.pool: list[dict] = []  # {"id", "state"}
        self.models = [copy.deepcopy(base_model).eval() for _ in range(groups)]
        self.current = [None] * groups
        self.pfsp = pfsp
        self.record = {}  # id -> [bot wins, games]
        self.fixed = set()  # groups whose opponent never changes (--human-opponent)

    def fix(self, g, model, tag):
        """Group g always plays this model (frozen, outside the pool)."""
        self.models[g].load_state_dict(model.state_dict())
        self.current[g] = tag
        self.fixed.add(g)

    def result(self, g, won):
        """A finished game against the opponent of group g (won: the learning bot won)."""
        r = self.record.setdefault(self.current[g], [0, 0])
        r[0] += int(won)
        r[1] += 1

    def weight(self, pid):
        """PFSP (AlphaStar): weight (1 - p)², p = the bot's win rate against this opponent (smoothed)."""
        w, n = self.record.get(pid, [0, 0])
        p = (w + 1) / (n + 2)
        return max(0.02, (1 - p) ** 2)

    def add(self, model, tag):
        self.pool.append({"id": tag, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}})
        if len(self.pool) > self.size:
            # always keep the imitation bot (the first one)
            self.pool.pop(1)

    def shuffle(self):
        for g, m in enumerate(self.models):
            if g in self.fixed:
                continue
            if self.pfsp and len(self.pool) > 1:
                pick = random.choices(self.pool, weights=[self.weight(p["id"]) for p in self.pool])[0]
            else:
                pick = random.choice(self.pool)
            if self.current[g] != pick["id"]:
                m.load_state_dict(pick["state"])
                self.current[g] = pick["id"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="checkpoint of the imitation bot (train/bc.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--install", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--rollout", type=int, default=64, help="decisions per iteration and per game")
    ap.add_argument("--self-frac", type=float, default=0.5)
    ap.add_argument("--groups", type=int, default=4, help="pool opponents played at the same time")
    ap.add_argument("--pool-size", type=int, default=10)
    ap.add_argument("--pool-every", type=int, default=50)
    ap.add_argument("--opp-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--kl-iters", type=int, default=2000)
    ap.add_argument("--shaping", type=float, default=1.0, help="weight of the tower damage bonus at the start")
    ap.add_argument("--shaping-iters", type=int, default=3000)
    ap.add_argument("--shaping-floor", type=float, default=0.0,
                    help="the tower damage bonus never goes below this (0 = it fades out). In 3 runs in a row, "
                         "the bot got worse as soon as it reached 0")
    ap.add_argument("--max-entities", type=int, default=96)
    ap.add_argument("--level", type=int, default=16, help="card and tower level (0 = the crforge default)")
    ap.add_argument("--plain", action="store_true", help="without evolutions or heroes")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--keep-every", type=int, default=250, help="keep a numbered copy")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--iters", type=int, default=0, help="max number of iterations (0 = no limit)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--value-warmup", type=int, default=20, help="iterations where only the critic learns")
    ap.add_argument("--kl-floor", type=float, default=0.01, help="the KL penalty never goes below this")
    ap.add_argument("--eval-every", type=int, default=50, help="evaluation against the imitation bot and the scripted bot")
    ap.add_argument("--eval-games", type=int, default=64)
    ap.add_argument("--eval-envs", type=int, default=32)
    ap.add_argument("--eval-threads", type=int, default=4)
    # -- real game conditions (PC bridge) --
    ap.add_argument("--levels", default="", help="levels drawn at random per game and per side, e.g. 9,10,11,11,11,12,13,14,16")
    ap.add_argument("--random-slots", action="store_true",
                    help="2.6 evolutions and hero present or not, at random per game and per side")
    ap.add_argument("--remix-every", type=int, default=50, help="new levels and forms every N iterations")
    ap.add_argument("--obs-noise", default="", help="input degraded like on the PC: unit_drop,noise,play_miss, e.g. 0.2,0.3,0.15")
    ap.add_argument("--action-delay", type=int, default=0, help="decisions of delay between the choice and the play (0 or 1)")
    ap.add_argument("--pfsp", action="store_true", help="older versions the bot beats less often come back more often (AlphaStar)")
    ap.add_argument("--privileged-critic", action="store_true",
                    help="during training, the critic also sees the opponent's header (hand, elixir)")
    ap.add_argument("--human-opponent", default="",
                    help="checkpoint of the real imitation bot: one opponent group always plays it (1/--groups of the "
                         "games against the pool). Without it, the 'bc' entry of the pool is the starting model, not the imitation bot")
    ap.add_argument("--moving-anchor", type=float, default=0,
                    help="if the bot beats the anchor (the reference model of the KL penalty) with at least this score in "
                         "the evaluation (e.g. 0.6), it becomes the new anchor (0 = the anchor stays the starting model)")
    args = ap.parse_args()
    args.tables = str(Path(args.tables).expanduser())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    t_start = time.time()
    deadline = t_start + args.max_hours * 3600 if args.max_hours else None

    # -- models ------------------------------------------------------------------------------------
    model, init_ck = load(args.init, dev)
    metadata = init_ck["metadata"]
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    opps = Opponents(model, args.groups, args.pool_size, dev, pfsp=args.pfsp)
    opps.add(model, "bc")
    it, steps = 0, 0
    ck = {}
    last = out / "ppo_last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        it, steps = ck["iter"], ck["steps"]
        opps.pool = ck["pool"]
        if "ref" in ck:  # the anchor may have moved (--moving-anchor)
            ref.load_state_dict(ck["ref"])
        print(f"resuming at iteration {it} ({steps} decisions)", flush=True)
    if args.human_opponent:
        # "human" opponent: the imitation bot plays like the replays. reel_it0400 beat it1450 61% of the
        # time, but did no better than it against the imitation bot (79.5% vs 81.5%, 800 games)
        human, _ = load(args.human_opponent, dev)
        opps.fix(0, human.eval(), "human")
        del human
    opps.shuffle()

    # -- games -------------------------------------------------------------------------------------
    v = VectorBattle(args.envs, args.threads, install=args.install, tables=args.tables, config=LadderConfig())
    s = v.schema
    if v.vocabulary_hash != metadata["vocabulary"]["hash"]:
        sys.exit("the tables do not match the model (different vocabulary)")
    v.write_metadata(out / "metadata.json")
    setup = None
    if args.level:
        setup = SideSetup(king_level=args.level, tower_level=args.level, card_levels=[args.level] * 8)
    slots = None if args.plain else SLOTS_26
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    rng_mix = random.Random(args.seed + 99)

    def mixed_reset(it_now):
        """One game per environment, each side with its own random level and forms."""
        seeds = [args.seed * 1_000_003 + it_now * 7919 + i for i in range(args.envs)]
        if not levels and not args.random_slots:
            v.reset(seeds, DECK_26, DECK_26, slots, slots, setup, setup)
            return
        sl, su = [[], []], [[], []]
        for _ in range(args.envs):
            for side in (0, 1):
                lvl = rng_mix.choice(levels) if levels else (args.level or 16)
                su[side].append(SideSetup(king_level=lvl, tower_level=lvl, card_levels=[lvl] * 8))
                if args.random_slots:
                    f = [0] * 8
                    f[DECK_26.index("Musketeer")] = 2 if rng_mix.random() < 0.5 else 0
                    f[DECK_26.index("Cannon")] = 1 if rng_mix.random() < 0.5 else 0
                    f[DECK_26.index("Skeletons")] = 1 if rng_mix.random() < 0.5 else 0
                    sl[side].append(f)
                else:
                    sl[side].append(list(slots) if slots else [0] * 8)
        v.reset(seeds, [DECK_26] * args.envs, [DECK_26] * args.envs, sl[0], sl[1], su[0], su[1])
    mixed_reset(it)

    n_self = int(args.envs * args.self_frac)
    le, ls, oe, os_, og = [], [], [], [], []  # bot seats / opponent seats and their group
    for i in range(args.envs):
        if i < n_self:
            le += [i, i]
            ls += [0, 1]
        else:
            seat = i % 2
            le.append(i)
            ls.append(seat)
            oe.append(i)
            os_.append(1 - seat)
            og.append((i - n_self) % args.groups)
    le, ls = np.array(le), np.array(ls)
    oe, os_, og = np.array(oe, dtype=int), np.array(os_, dtype=int), np.array(og, dtype=int)
    vs_pool = np.array([i >= n_self for i in le])
    NL = len(le)
    W = HEADER + args.max_entities * ENT
    ci = s.header["entity_count"][0]
    h_own = [s.header[k][0] for k in ("own_king_hp", "own_left_hp", "own_right_hp")]
    h_own_max = [s.header[k][0] for k in ("own_king_max_hp", "own_left_max_hp", "own_right_max_hp")]
    h_en = [s.header[k][0] for k in ("enemy_king_hp", "enemy_left_hp", "enemy_right_hp")]
    h_en_max = [s.header[k][0] for k in ("enemy_king_max_hp", "enemy_left_max_hp", "enemy_right_max_hp")]
    I_RESET, I_FAULT, I_WINNER = s.info["reset"], s.info["fault"], s.info["winner"]
    T = args.rollout

    log = open(out / "ppo_log.jsonl", "a")
    # regular evaluations: against the imitation bot (fixed) and the crforge scripted bot
    ev = VectorBattle(args.eval_envs, args.eval_threads, install=args.install, tables=args.tables, config=LadderConfig())
    ev_ci = ev.schema.header["entity_count"][0]
    opp_bc = model_policy(ref, dev, ev_ci)
    opp_scripted = make_opponent("scripted", ev, dev, ev_ci)
    best = ck.get("best", -1.0) if args.resume and last.exists() else -1.0

    def evaluate(tag):
        model.eval()
        me = model_policy(model, dev, ev_ci)
        r_bc = run_games(ev, me, opp_bc, args.eval_games, seed=10_000 + it)
        r_sc = run_games(ev, me, opp_scripted, args.eval_games, seed=20_000 + it)
        rec = {"eval": tag, "iter": it, "steps": steps, "vs_bc": r_bc["score"], "vs_scripted": r_sc["score"],
               "len_bc": r_bc["mean_length"], "len_scripted": r_sc["mean_length"]}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"  eval {tag} (it {it}): vs imitation {r_bc['score']:.0%}, vs scripted bot "
              f"{r_sc['score']:.0%} ({args.eval_games} games each)", flush=True)
        return rec

    if args.eval_every and it == 0:
        evaluate("start")

    buf_obs = torch.zeros((T, NL, W), dtype=torch.int32, device=dev)
    buf_mask = torch.zeros((T, NL, s.actions), dtype=torch.bool, device=dev)
    buf_act = torch.zeros((T, NL), dtype=torch.long, device=dev)
    buf_logp = torch.zeros((T, NL), device=dev)
    buf_val = torch.zeros((T, NL), device=dev)
    buf_rew = torch.zeros((T, NL), device=dev)
    buf_done = torch.zeros((T, NL), device=dev)
    buf_valid = torch.zeros((T, NL), device=dev)
    seeall = args.privileged_critic
    buf_priv = torch.zeros((T, NL, HEADER), dtype=torch.int32, device=dev) if seeall else None
    opp_seat = 1 - ls  # the opposite seat, for the privileged critic
    amp = autocast(dev)
    ep_len = np.zeros(args.envs, dtype=np.int64)

    noise = [float(x) for x in args.obs_noise.split(",")] if args.obs_noise else None
    E_KIND, E_X, E_Y, E_DX, E_DY = (s.entity[k] for k in ("kind", "x", "y", "dx", "dy"))
    play_fields = [s.header[k] for k in ("enemy_play_card", "enemy_play_cast", "enemy_play_form", "enemy_play_ticks",
                                         "enemy_play_x", "enemy_play_y", "enemy_play_kind", "enemy_play_flags",
                                         "enemy_play_point_ticks")]
    h_enemy_elixir = s.header["enemy_elixir"][0]
    rng_noise = np.random.default_rng(args.seed + 7)

    def corrupt(ob, n_ent):
        """The input as the PC kit gives it: units missed by the vision, rough positions,
        opponent plays missed or shifted, the opponent's elixir miscounted."""
        if noise is None:
            return ob
        p_drop, sigma, p_play = noise
        ob = ob.copy()
        n = len(ob)
        ent = ob[:, HEADER: HEADER + n_ent * ENT].reshape(n, n_ent, ENT)
        count = ob[:, ci].clip(0, n_ent)
        idx = np.arange(n_ent)[None, :]
        unit = (idx >= 6) & (idx < count[:, None]) & (ent[..., E_KIND] == 1)
        drop = unit & (rng_noise.random((n, n_ent)) < p_drop)
        move = unit & ~drop
        ent[..., E_X] = np.where(move, (ent[..., E_X] + rng_noise.normal(0, sigma * 1000, (n, n_ent))).clip(0, 18000), ent[..., E_X])
        ent[..., E_Y] = np.where(move, (ent[..., E_Y] + rng_noise.normal(0, sigma * 1000, (n, n_ent))).clip(0, 32000), ent[..., E_Y])
        for f in (E_DX, E_DY):
            ent[..., f] = np.where(move, ent[..., f] + rng_noise.normal(0, sigma * 300, (n, n_ent)), ent[..., f])
        if drop.any():  # remove the missed rows and keep the order of the others
            order = np.argsort(drop, axis=1, kind="stable")
            ent[:] = np.take_along_axis(ent, order[..., None], axis=1)
            new_count = count - drop.sum(1)
            ent[idx >= new_count[:, None]] = 0  # nothing after the last kept unit
            ob[:, ci] = new_count
        ob[:, HEADER: HEADER + n_ent * ENT] = ent.reshape(n, -1)
        # opponent plays: some are missed (removed from the list, the next ones move up), the others
        # are shifted in time and space
        rows = len(play_fields[0])
        shown = ob[:, play_fields[0]] >= 0
        miss = shown & (rng_noise.random((n, rows)) < p_play)
        order = np.argsort(miss, axis=1, kind="stable")
        kept = (~miss).sum(1)
        blank = np.arange(rows)[None, :] >= kept[:, None]
        for j, cols in enumerate(play_fields):
            cols = np.array(cols)
            vals = np.take_along_axis(ob[:, cols], order, axis=1)
            vals = np.where(blank, -1 if j in (0, 1) else 0, vals)
            live = ~blank & (vals != 0)
            if j == 3:
                vals = np.where(live, (vals + rng_noise.integers(-6, 7, vals.shape)).clip(1), vals)
            elif j in (4, 5):
                vals = np.where(live & (vals > 0), vals + rng_noise.normal(0, 600, vals.shape).astype(np.int64), vals)
            ob[:, cols] = vals
        ob[:, h_enemy_elixir] = (ob[:, h_enemy_elixir] + rng_noise.normal(0, 7000, n)).clip(0, 100000)
        return ob

    delayed = np.zeros(NL, dtype=np.int64)  # the play chosen at the previous decision (reaction delay)

    def towers(o):
        own = o[:, h_own].clip(min=0).sum(1) / np.maximum(1, o[:, h_own_max].sum(1))
        en = o[:, h_en].clip(min=0).sum(1) / np.maximum(1, o[:, h_en_max].sum(1))
        return own, en

    def policy_actions(m, obs_np, mask_np, priv_np=None):
        o = torch.from_numpy(obs_np).to(dev, non_blocking=True)
        k = torch.from_numpy(mask_np).to(dev, non_blocking=True)
        pv = torch.from_numpy(priv_np).to(dev, non_blocking=True) if priv_np is not None else None
        with torch.no_grad(), amp:
            f, t, val = m(o, k, pv)
            a = sample(f, t)
            lp = log_prob(f, t, a)
        return o, k, a, lp.float(), val.float()

    while True:
        if args.iters and it >= args.iters:
            break
        if deadline and time.time() > deadline:
            print("time limit reached, clean stop", flush=True)
            break
        t_iter = time.time()
        if (levels or args.random_slots) and it > 0 and it % args.remix_every == 0:
            mixed_reset(it)  # new levels and forms (the games in progress end here)
            ep_len[:] = 0
            delayed[:] = 0
        shaping = max(args.shaping_floor, linear(it, args.shaping, args.shaping_iters))
        kl_coef = max(args.kl_floor, linear(it, args.kl_coef, args.kl_iters)) if args.kl_coef > 0 else 0.0
        stats = {"wins_pool": 0, "games_pool": 0, "wins_self": 0, "games_self": 0, "ep_len": [], "faults": 0}
        model.eval()
        tm = {"policy": 0.0, "sim": 0.0, "update": 0.0}  # where the time goes
        for t in range(T):
            obs_all, mask_all = v.observations, v.masks
            n_ent = int(min(args.max_entities, max(1, obs_all[..., ci].max())))
            w = HEADER + n_ent * ENT
            ob = np.ascontiguousarray(obs_all[le, ls, :w])
            mk = mask_all[le, ls].astype(bool)
            own0, en0 = towers(ob)
            ob = corrupt(ob, n_ent)
            t_pol = time.perf_counter()
            pv = np.ascontiguousarray(obs_all[le, opp_seat, :HEADER]) if seeall else None
            o, k, a, lp, val = policy_actions(model, ob, mk, pv)
            if seeall:
                buf_priv[t] = torch.from_numpy(pv).to(dev)
            buf_obs[t].zero_()
            buf_obs[t, :, :w] = o
            buf_mask[t] = k
            buf_act[t] = a
            buf_logp[t] = lp
            buf_val[t] = val
            a_np = a.cpu().numpy()
            if args.action_delay:
                # the play goes out one decision later, if it is still allowed (elixir, hand)
                go = np.where(mk[np.arange(NL), delayed], delayed, 0)
                v.actions[le, ls] = go
                delayed = a_np.copy()
            else:
                v.actions[le, ls] = a_np
            for g, m in enumerate(opps.models):
                sel = og == g
                if not sel.any():
                    continue
                obo = np.ascontiguousarray(obs_all[oe[sel], os_[sel], :w])
                mko = mask_all[oe[sel], os_[sel]].astype(bool)
                _, _, ao, _, _ = policy_actions(m, obo, mko)
                v.actions[oe[sel], os_[sel]] = ao.cpu().numpy()
            tm["policy"] += time.perf_counter() - t_pol
            t_sim = time.perf_counter()
            v.step()
            tm["sim"] += time.perf_counter() - t_sim
            steps += NL
            ep_len += 1
            info = v.infos
            reset = info[le, I_RESET] != 0
            done = (v.dones[le] != 0) | reset
            if args.action_delay:
                delayed[done] = 0
            rew = v.rewards[le, ls].astype(np.float32)
            if shaping > 0:
                ob1 = v.observations[le, ls, :HEADER]
                own1, en1 = towers(ob1)
                shaped = shaping * ((en0 - en1) - (own0 - own1))
                rew = rew + np.where(reset, 0.0, shaped).astype(np.float32)
            buf_rew[t] = torch.from_numpy(rew).to(dev)
            buf_done[t] = torch.from_numpy(done.astype(np.float32)).to(dev)
            buf_valid[t] = torch.from_numpy((~reset).astype(np.float32)).to(dev)
            finished = np.flatnonzero((v.dones != 0) & (info[:, I_RESET] == 0))
            for i in finished:
                stats["ep_len"].append(int(ep_len[i]))
                ep_len[i] = 0
                stats["faults"] += int(info[i, I_FAULT] != 0)
                winner = int(info[i, I_WINNER])
                if i < n_self:
                    stats["games_self"] += 1
                    stats["wins_self"] += int(winner == 0)
                else:
                    stats["games_pool"] += 1
                    stats["wins_pool"] += int(winner == i % 2)
                    opps.result((i - n_self) % args.groups, winner == i % 2)
        # value of the last observation
        obs_all, mask_all = v.observations, v.masks
        n_ent = int(min(args.max_entities, max(1, obs_all[..., ci].max())))
        w = HEADER + n_ent * ENT
        pv = np.ascontiguousarray(obs_all[le, opp_seat, :HEADER]) if seeall else None
        _, _, _, _, last_val = policy_actions(model, np.ascontiguousarray(obs_all[le, ls, :w]), mask_all[le, ls].astype(bool), pv)

        # -- GAE ---------------------------------------------------------------------------------
        adv = torch.zeros_like(buf_rew)
        gae = torch.zeros(NL, device=dev)
        for t in reversed(range(T)):
            nxt = last_val if t == T - 1 else buf_val[t + 1]
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + args.gamma * nxt * nonterm - buf_val[t]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv[t] = gae
        ret = adv + buf_val

        # -- PPO update --------------------------------------------------------------------------
        t_upd = time.perf_counter()
        model.train()
        flat = lambda x: x.reshape(T * NL, *x.shape[2:])  # noqa: E731
        f_obs, f_mask, f_act, f_logp = flat(buf_obs), flat(buf_mask), flat(buf_act), flat(buf_logp)
        f_adv, f_ret, f_valid = flat(adv), flat(ret), flat(buf_valid)
        f_priv = flat(buf_priv) if seeall else None
        idx = torch.nonzero(f_valid > 0).squeeze(1)
        a_valid = f_adv[idx]
        f_adv = (f_adv - a_valid.mean()) / (a_valid.std() + 1e-8)
        m_stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl_ref": 0.0, "clip": 0.0, "approx_kl": 0.0, "n": 0}
        for _ in range(args.epochs):
            perm = idx[torch.randperm(len(idx), device=dev)]
            for b in range(0, len(perm), args.minibatch):
                j = perm[b: b + args.minibatch]
                ob = f_obs[j]
                n_ent = int(min(args.max_entities, max(1, ob[:, ci].max().item())))
                ob = ob[:, : HEADER + n_ent * ENT]
                with amp:
                    first, tile, value = model(ob, f_mask[j], f_priv[j] if seeall else None)
                    lp = log_prob(first, tile, f_act[j])
                    ratio = torch.exp(lp - f_logp[j])
                    A = f_adv[j]
                    pg = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
                    vf = ((value - f_ret[j]) ** 2).mean()
                    ent = entropy(first, tile).mean()
                    loss = pg + args.vf_coef * vf - args.ent_coef * ent
                    if kl_coef > 0:
                        with torch.no_grad():
                            rf, rt, _ = ref(ob, f_mask[j])
                        klr = kl(first, tile, rf, rt).mean()
                        loss = loss + kl_coef * klr
                    else:
                        klr = torch.zeros((), device=dev)
                    if it < args.value_warmup:  # warmup: only the critic learns
                        loss = args.vf_coef * vf
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if it < args.value_warmup:
                    for name, prm in model.named_parameters():
                        if not (name.startswith("value_head") or name.startswith("priv_proj")):
                            prm.grad = None
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    m_stats["pg"] += pg.item()
                    m_stats["vf"] += vf.item()
                    m_stats["ent"] += ent.item()
                    m_stats["kl_ref"] += klr.item()
                    m_stats["clip"] += ((ratio - 1).abs() > args.clip).float().mean().item()
                    m_stats["approx_kl"] += ((ratio - 1) - torch.log(ratio)).mean().item()
                    m_stats["n"] += 1
        tm["update"] = time.perf_counter() - t_upd
        it += 1
        n = max(1, m_stats.pop("n"))
        rec = {k: val / n for k, val in m_stats.items()}
        rec.update(iter=it, steps=steps, sps=T * NL / (time.time() - t_iter), shaping=shaping, kl_coef=kl_coef,
                   win_vs_pool=stats["wins_pool"] / max(1, stats["games_pool"]), games_pool=stats["games_pool"],
                   win_self_seat0=stats["wins_self"] / max(1, stats["games_self"]), games_self=stats["games_self"],
                   ep_len=float(np.mean(stats["ep_len"])) if stats["ep_len"] else 0.0, faults=stats["faults"],
                   opponents=list(opps.current), pool=[p["id"] for p in opps.pool], hours=(time.time() - t_start) / 3600,
                   time_total=time.time() - t_iter, time_sim=tm["sim"], time_policy=tm["policy"], time_update=tm["update"])
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % args.log_every == 0 or it == 1:
            print(f"it {it} | {steps / 1e6:.1f} M decisions | {rec['sps']:.0f}/s | vs pool {rec['win_vs_pool']:.1%} "
                  f"({rec['games_pool']} games) | entropy {rec['ent']:.2f} | kl ref {rec['kl_ref']:.3f} "
                  f"| vf {rec['vf']:.3f} | length {rec['ep_len']:.0f} dec. | time: simulator {tm['sim'] / rec['time_total']:.0%}, "
                  f"network {tm['policy'] / rec['time_total']:.0%}, update {tm['update'] / rec['time_total']:.0%}", flush=True)
        if args.eval_every and it % args.eval_every == 0:
            rec_eval = evaluate("regular")
            if rec_eval["vs_bc"] > best:
                best = rec_eval["vs_bc"]
                save(out / "ppo_best.pt", model, metadata, {"iter": it, "steps": steps, "eval": rec_eval})
                print(f"  new best model: {best:.0%} vs imitation -> ppo_best.pt", flush=True)
            if args.moving_anchor and rec_eval["vs_bc"] >= args.moving_anchor:
                # the bot has clearly passed the anchor: it becomes the anchor (KL and evaluations are
                # measured against it from now on). Without this, the KL penalty keeps it close to the
                # starting model and it stalls (in one run, the best model was at iteration 400 of 2,010).
                ref.load_state_dict(model.state_dict())
                save(out / f"ppo_anchor_it{it:06d}.pt", model, metadata, {"iter": it, "steps": steps, "eval": rec_eval})
                save(out / "ppo_best.pt", model, metadata, {"iter": it, "steps": steps, "eval": rec_eval})
                opps.add(model, f"anchor{it}")
                best = 0.5
                log.write(json.dumps({"anchor": it, "vs_previous_anchor": rec_eval["vs_bc"]}) + "\n")
                log.flush()
                print(f"  new anchor at iteration {it} ({rec_eval['vs_bc']:.0%} vs the previous anchor) "
                      f"-> ppo_anchor_it{it:06d}.pt", flush=True)
        if it % args.pool_every == 0:
            opps.add(model, f"it{it}")
        if it % args.opp_every == 0:
            opps.shuffle()
        if it % args.save_every == 0:
            extra = {"opt": opt.state_dict(), "iter": it, "steps": steps, "pool": opps.pool, "args": vars(args), "best": best,
                     "ref": ref.state_dict()}
            save(last, model, metadata, extra)
            if it % args.keep_every == 0:
                save(out / f"ppo_it{it:06d}.pt", model, metadata, {"iter": it, "steps": steps})
    extra = {"opt": opt.state_dict(), "iter": it, "steps": steps, "pool": opps.pool, "args": vars(args), "best": best,
             "ref": ref.state_dict()}
    save(last, model, metadata, extra)
    v.close()
    ev.close()
    print(f"done at iteration {it}, {steps} decisions. Last model: {last}", flush=True)


if __name__ == "__main__":
    main()
