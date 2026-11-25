"""Imitation: learn to play like the 2.6 players of IL_Replay.

Reads the rows written by sim/il_to_crforge.py (observation, mask, play), and learns:
- the play (do nothing, which card, which tile) by maximum likelihood;
- the value (who wins the real game), so that PPO starts with a critic that is already warm.

The converter kept only part of the "do nothing" decisions (--keep-wait). We train on this
mix, then correct the "wait" choice with a bias of log(1 / keep_wait), stored in the model:
the bot then plays at the same pace as the players.

Usage:
  python train/bc.py --data data/crforge_bc --out runs/bc --device cuda --epochs 6

Bigger network (about 4.6 M parameters instead of 1.2 M):
  --d-model 256 --heads 8 --layers 4 --trunk 512
Pretraining on all games, then 2.6 fine-tuning starting from the model it gives:
  python train/bc.py --data data/il_all --out runs/bc_all ... ; python train/bc.py --data data/crforge_bc --init runs/bc_all/bc_best.pt --lr 1e-4 ...
Validation games are chosen by their id (replay_tag), so they are the same from one dataset to another.
"""

from __future__ import annotations

import argparse
import glob
import math
import zlib
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.policy import autocast, ENT, HEADER, Layout, Policy, PolicyConfig, load, log_prob, save, split_action, wait_bias_for  # noqa: E402


def results_by_match(data_dirs):
    """game index -> +1 if the bottom team (seat 0) won the real game, -1 otherwise."""
    out = {}
    for d in data_dirs:
        for f in glob.glob(str(Path(d) / "matches_*.jsonl")):
            for line in open(f):
                st = json.loads(line)
                out[st["index"]] = {"victory": 1.0, "defeat": -1.0}.get(st["result"], 0.0)
    return out


def val_matches(data_dirs, val_mod):
    """indices of the validation games: 1 in val_mod, chosen by replay_tag (stable from one dataset to another)."""
    out = set()
    for d in data_dirs:
        for f in glob.glob(str(Path(d) / "matches_*.jsonl")):
            for line in open(f):
                st = json.loads(line)
                if zlib.crc32(str(st["tag"]).encode()) % val_mod == 0:
                    out.add(st["index"])
    return out


class Shards:
    """Loads the row files in groups, filters, shuffles, and serves batches."""

    def __init__(self, files, args, results, layout, train: bool, val_set=None):
        self.files, self.args, self.results, self.train = list(files), args, results, train
        self.val_set = np.array(sorted(val_set), dtype=np.int64) if val_set is not None else None
        self.width = HEADER + args.max_entities * ENT
        self.count_index = layout.h("entity_count")[0]

    def _load(self, group):
        obs, mask, act, val = [], [], [], []
        for f in group:
            d = np.load(f)
            match = d["match"]
            keep = d["dropped"] <= self.args.max_dropped
            if self.val_set is not None:
                split = np.isin(match, self.val_set)
            else:
                split = (match % self.args.val_mod) == 0
            keep &= ~split if self.train else split
            if not keep.any():
                continue
            o = d["obs"][keep]
            obs.append(o[:, : self.width])
            mask.append(np.unpackbits(d["mask"][keep], axis=1)[:, :2307])
            act.append(d["action"][keep])
            sign = np.where(d["seat"][keep] == 0, 1.0, -1.0)
            val.append(sign * np.array([self.results.get(int(m), 0.0) for m in match[keep]]))
        if not obs:
            return None
        return (np.concatenate(obs), np.concatenate(mask), np.concatenate(act),
                np.concatenate(val).astype(np.float32))

    def __iter__(self):
        files = list(self.files)
        rng = np.random.default_rng()
        if self.train:
            rng.shuffle(files)
        groups = [files[i: i + self.args.group] for i in range(0, len(files), self.args.group)]
        q: queue.Queue = queue.Queue(maxsize=2)

        def producer():
            for g in groups:
                q.put(self._load(g))
            q.put(StopIteration)

        threading.Thread(target=producer, daemon=True).start()
        bs = self.args.batch
        while True:
            item = q.get()
            if item is StopIteration:
                return
            if item is None:
                continue
            obs, mask, act, val = item
            order = rng.permutation(len(act)) if self.train else np.arange(len(act))
            for i in range(0, len(order), bs):
                j = order[i: i + bs]
                yield (torch.from_numpy(obs[j]), torch.from_numpy(mask[j]),
                       torch.from_numpy(act[j]), torch.from_numpy(val[j]))


def lr_schedule(total_steps, budget_s, warm=0.05, start=0.04, end=1e-3):
    """Warmup over 5%, then cosine decay. Progress is the larger of the two: steps done, or time
    spent out of --max-hours. So the model ends with a small learning rate even if time runs out
    before the last epoch, and the schedule does not crash if the step estimate is too low."""
    t0 = time.time()

    def f(step):
        p = step / max(1, total_steps)
        if budget_s:
            p = max(p, (time.time() - t0) / budget_s)
        p = min(1.0, p)
        if p < warm:
            return start + (1 - start) * p / warm
        return end + (1 - end) * 0.5 * (1 + math.cos(math.pi * (p - warm) / (1 - warm)))
    return f


def run_epoch(model, shards, args, opt=None, sched=None, deadline=None):
    train = opt is not None
    model.train(train)
    dev = args.device
    tot = {"n": 0, "nll": 0.0, "vloss": 0.0, "plays": 0, "first_ok": 0, "tile_ok": 0, "waits": 0, "wait_ok": 0}
    amp = autocast(dev)
    t0, steps = time.time(), 0
    for obs, mask, act, val in shards:
        obs, mask = obs.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
        act, val = act.to(dev, non_blocking=True), val.to(dev, non_blocking=True)
        with torch.set_grad_enabled(train), amp:
            first_logp, tile_logp, value = model(obs, mask)
            nll = -log_prob(first_logp, tile_logp, act).mean()
            vloss = ((value - val) ** 2).mean()
            loss = nll + args.value_coef * vloss
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        with torch.no_grad():
            first, slot, tile, is_play = split_action(act)
            pf = first_logp.argmax(-1)
            pt = tile_logp[torch.arange(len(act), device=act.device), slot].argmax(-1)
            n = len(act)
            tot["n"] += n
            tot["nll"] += nll.item() * n
            tot["vloss"] += vloss.item() * n
            tot["plays"] += int(is_play.sum())
            tot["first_ok"] += int(((pf == first) & is_play).sum())
            tot["tile_ok"] += int(((pt == tile) & is_play).sum())
            tot["waits"] += int((act == 0).sum())
            tot["wait_ok"] += int(((pf == 0) & (act == 0)).sum())
        steps += 1
        if train and steps % args.log_every == 0:
            print(f"  step {steps} | nll {tot['nll'] / tot['n']:.3f} | card {tot['first_ok'] / max(1, tot['plays']):.1%} "
                  f"| tile {tot['tile_ok'] / max(1, tot['plays']):.1%} | {tot['n'] / (time.time() - t0):.0f} rows/s "
                  f"| lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        if deadline and time.time() > deadline:
            break
    n = max(1, tot["n"])
    return {"rows": tot["n"], "nll": tot["nll"] / n, "value_mse": tot["vloss"] / n,
            "card_acc": tot["first_ok"] / max(1, tot["plays"]), "tile_acc": tot["tile_ok"] / max(1, tot["plays"]),
            "wait_acc": tot["wait_ok"] / max(1, tot["waits"]), "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="output folder(s) of sim/il_to_crforge.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.25)
    ap.add_argument("--max-dropped", type=int, default=5, help="skip the rows that come after more than N dropped plays")
    ap.add_argument("--max-entities", type=int, default=64)
    ap.add_argument("--group", type=int, default=4, help="files loaded together")
    ap.add_argument("--val-mod", type=int, default=20, help="1 game in N goes to validation")
    ap.add_argument("--keep-wait", type=float, default=None, help="read from convert_args.json if not given")
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="for a quick test")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default="", help="start from the weights of this model (fine-tuning after pretraining)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--trunk", type=int, default=256)
    ap.add_argument("--val-by-index", action="store_true",
                    help="old choice of the validation games (index %% val_mod), to compare with older results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    meta_path = Path(args.data[0]) / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    if args.keep_wait is None:
        conv = Path(args.data[0]) / "convert_args.json"
        args.keep_wait = json.loads(conv.read_text())["keep_wait"] if conv.exists() else 0.1
    layout = Layout(metadata)
    files = sorted(f for d in args.data for f in glob.glob(str(Path(d) / "rows_*.npz")))
    if not files:
        sys.exit("no rows_*.npz file")
    results = results_by_match(args.data)
    print(f"{len(files)} files, {len(results)} games, keep_wait {args.keep_wait}", flush=True)

    if args.init:
        model, ck0 = load(args.init, args.device)  # its own size and weights
        model.wait_bias.zero_()
        if ck0["metadata"]["vocabulary"]["hash"] != metadata["vocabulary"]["hash"]:
            sys.exit("the starting model does not have the same vocabulary as the data")
        print(f"starting from {args.init}", flush=True)
    else:
        cfg = PolicyConfig(d_model=args.d_model, heads=args.heads, layers=args.layers, trunk=args.trunk)
        model = Policy(layout, cfg).to(args.device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters on {args.device}", flush=True)
    vset = None if args.val_by_index else val_matches(args.data, args.val_mod)
    if vset is not None:
        print(f"{len(vset)} validation games (chosen by replay_tag)", flush=True)
    train_set = Shards(files, args, results, layout, train=True, val_set=vset)
    val_set = Shards(files, args, results, layout, train=False, val_set=vset)
    # steps per epoch, estimated on the first file
    d0 = np.load(files[0])
    in_val = np.isin(d0["match"], np.array(sorted(vset))) if vset is not None else (d0["match"] % args.val_mod == 0)
    rows_est = int(((d0["dropped"] <= args.max_dropped) & ~in_val).sum()) * len(files)
    steps_per_epoch = max(1, rows_est // args.batch)
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule(total, args.max_hours * 3600))
    start_epoch, best = 0, float("inf")
    last = out / "bc_last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["state"])
        model.wait_bias.zero_()
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resuming at epoch {start_epoch}", flush=True)
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    bias = wait_bias_for(args.keep_wait)
    log = open(out / "bc_log.jsonl", "a")

    def save_with_bias(path, extra):
        model.wait_bias.fill_(bias)
        save(path, model, metadata, extra)
        model.wait_bias.zero_()

    for epoch in range(start_epoch, args.epochs):
        if args.max_steps_per_epoch:
            train_iter = (b for k, b in zip(range(args.max_steps_per_epoch), train_set))
        else:
            train_iter = train_set
        tr = run_epoch(model, train_iter, args, opt, sched, deadline)
        va = run_epoch(model, val_set, args)
        rec = {"epoch": epoch, "train": tr, "val": va}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"epoch {epoch} | train nll {tr['nll']:.3f} | val nll {va['nll']:.3f} card {va['card_acc']:.1%} "
              f"tile {va['tile_acc']:.1%} wait {va['wait_acc']:.1%} value {va['value_mse']:.3f} | {tr['seconds']:.0f}s", flush=True)
        extra = {"opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch, "best": min(best, va["nll"]),
                 "keep_wait": args.keep_wait, "val": va}
        save_with_bias(last, extra)
        if va["nll"] < best:
            best = va["nll"]
            save_with_bias(out / "bc_best.pt", extra)
        if deadline and time.time() > deadline:
            print("time limit reached, clean stop", flush=True)
            break
    print(f"done. best model: {out / 'bc_best.pt'} (val nll {best:.3f})", flush=True)


if __name__ == "__main__":
    main()
