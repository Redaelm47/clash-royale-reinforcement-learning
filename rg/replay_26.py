"""Replays the 2.6 matches of IL_Replay in RoyaleGym and writes demos (RoyaleImitate shards).

Differences with royaleimitate.from_replays:
- only matches where one deck is exactly the 2.6 are kept (-ev1 / -hero suffixes removed);
- only the 2.6 seat is recorded (both seats in a mirror);
- each process takes a subset of the 52 chunks, so N jobs can run in parallel.

Examples:
    python replay_26.py --data-dir IL_Replay --shard 0 --num-shards 20 --out runs/demos/shard00
    python replay_26.py --chunks 0-9 --out runs/demos/a          # downloads the chunks from HF
    python replay_26.py merge --out runs/demos/all runs/demos/shard*
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from env_26 import HOG_26, find_shards, learner_26

N_CHUNKS = 52  # replays/part-000000 to part-000051, 5,000 matches each
HOG = frozenset(HOG_26)


def parse_chunks(text):
    """'0-9,15,20-22' -> [0, ..., 9, 15, 20, 21, 22]"""
    out = []
    for piece in text.split(","):
        piece = piece.strip()
        if "-" in piece:
            a, b = piece.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif piece:
            out.append(int(piece))
    return sorted(set(out))


def chosen_chunks(args):
    if args.chunks:
        chunks = parse_chunks(args.chunks)
    else:
        chunks = list(range(N_CHUNKS))
    if args.num_shards > 1:
        chunks = chunks[args.shard :: args.num_shards]
    return chunks


def chunk_path(index, data_dir, cache_dir):
    """The parquet file of a chunk: from --data-dir, otherwise downloaded from Hugging Face."""
    name = f"replays/part-{index:06d}.parquet"
    if data_dir:
        path = Path(data_dir) / name
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        return path
    from huggingface_hub import hf_hub_download
    from royaleimitate.replays import DATASET

    return Path(hf_hub_download(DATASET, name, repo_type="dataset", cache_dir=cache_dir))


def payloads_26(path):
    """(row number, match) for the matches of a chunk that have at least one 2.6 deck."""
    import pyarrow.parquet as pq
    from royaleimitate.replays import CARD_NAMES, base_key

    def is_26(players):
        if len(players) != 1:
            return False
        return {CARD_NAMES.get(base_key(c["card_key"])) for c in players[0]["deck"]} == HOG

    seen = 0
    table = pq.ParquetFile(path)
    for batch in table.iter_batches(columns=["payload_json"], batch_size=256):
        for text in batch.column(0).to_pylist():
            seen += 1
            payload = json.loads(text)
            battle = payload["battle"]
            if is_26(battle["team"]["players"]) or is_26(battle["opponent"]["players"]):
                yield seen, payload


def pq_rows(path):
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def replay(args):
    from royaleimitate.replays import _AGENTS, _battle_env, _replay, match_plan
    from royaleimitate.shards import ShardContext, ShardWriter

    chunks = chosen_chunks(args)
    if not chunks:
        raise SystemExit("no chunk for this shard")
    out = Path(args.out)
    print(f"chunks {chunks} -> {out}", flush=True)

    learner = learner_26(out / "_learner", device="cpu")
    env = learner.build_env()
    battle = _battle_env(env)
    catalogue = frozenset(card.name for card in battle.engine.cards())
    context = ShardContext.of_config(learner.config)

    stats = Counter()
    failed = Counter()
    rows = plays = 0
    started = time.monotonic()
    replay_time = 0.0
    producer = {"tool": "rg/replay_26.py", "chunks": chunks, "deck": HOG_26}
    with ShardWriter(out, context, producer=producer) as writer:
        for chunk in chunks:
            if stats["replayed"] >= args.max_matches:
                break
            path = chunk_path(chunk, args.data_dir, args.hf_cache)
            seen = 0
            for seen, payload in payloads_26(path):
                if stats["replayed"] >= args.max_matches:
                    break
                stats["kept"] += 1
                plan = match_plan(payload, catalogue)
                if isinstance(plan, str):
                    failed[plan] += 1
                    continue
                seats = [s for s in (0, 1) if set(plan.decks[s]) == HOG]
                t0 = time.monotonic()
                result = _replay(env, battle, plan, seed=chunk * 100_000 + seen)
                replay_time += time.monotonic() - t0
                if isinstance(result, str):
                    failed[result] += 1
                    continue
                group = int(plan.tag[:15], 16)
                for agent, obs, action, step in result:
                    seat = _AGENTS.index(agent)
                    if seat not in seats:
                        continue
                    writer.add(obs, action=action, weight=1.0, group=group, seat=seat, tick=step)
                    rows += 1
                    plays += action != 0
                stats["replayed"] += 1
                stats["mirror"] += len(seats) == 2
                if stats["replayed"] % 50 == 0:
                    rate = replay_time / stats["replayed"]
                    print(f"  {stats['replayed']} replayed, {sum(failed.values())} failed, "
                          f"{rate:.2f} s/match", flush=True)
            else:
                seen = pq_rows(path)
            stats["scanned"] += seen
            print(f"chunk {chunk} done: {dict(stats)}", flush=True)
        writer.manifest.producer.update({
            "matches": stats["replayed"], "rows": rows, "play_rows": plays,
            "kept": stats["kept"], "failed": dict(failed),
        })
    close = getattr(env, "close", None)
    if close:
        close()
    shutil.rmtree(out / "_learner", ignore_errors=True)

    took = time.monotonic() - started
    summary = {
        "chunks": chunks,
        "matches_scanned": stats["scanned"],
        "matches_26_kept": stats["kept"],
        "matches_replayed": stats["replayed"],
        "mirror_matches": stats["mirror"],
        "matches_failed": sum(failed.values()),
        "failed_why": dict(failed.most_common()),
        "rows": rows,
        "play_rows": plays,
        "seconds_total": round(took, 1),
        "seconds_per_replayed_match": round(replay_time / max(1, stats["replayed"]), 3),
        "shards": str(out / context.engine_key),
    }
    (out / "stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def merge(args):
    """Merges several demo folders into one (hard links, no copy when possible)."""
    import msgspec
    from royaleimitate.shards import MANIFEST_NAME, PartInfo, ShardManifest

    sources = [find_shards(p) for p in args.sources]
    manifests = [msgspec.json.decode((s / MANIFEST_NAME).read_bytes(), type=ShardManifest)
                 for s in sources]
    first = manifests[0]
    for src, man in zip(sources, manifests):
        if man.engine_key != first.engine_key or man.context != first.context:
            raise SystemExit(f"{src} was not replayed on the same engine as {sources[0]}")
    dest = Path(args.out) / first.engine_key
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} already exists")
    dest.mkdir(parents=True, exist_ok=True)

    parts, flags, totals, failed = [], Counter(), Counter(), Counter()
    for src, man in zip(sources, manifests):
        for part in man.parts:
            name = f"part-{len(parts):05d}.npz"
            try:
                os.link(src / part.name, dest / name)
            except OSError:
                shutil.copy2(src / part.name, dest / name)
            parts.append(PartInfo(name=name, sha256=part.sha256, rows=part.rows,
                                  validation_rows=part.validation_rows))
        flags.update(man.flag_counts)
        for key in ("matches", "rows", "play_rows", "kept"):
            totals[key] += int(man.producer.get(key, 0))
        failed.update(man.producer.get("failed", {}))
    producer = {"tool": "rg/replay_26.py merge", "sources": [str(s) for s in sources],
                **totals, "failed": dict(failed)}
    merged = msgspec.structs.replace(
        first, parts=parts, producer=producer, flag_counts=dict(flags),
        rows=sum(p.rows for p in parts), validation_rows=sum(p.validation_rows for p in parts),
    )
    blob = msgspec.json.format(msgspec.json.encode(merged), indent=1)
    (dest / MANIFEST_NAME).write_bytes(blob)
    print(f"{len(sources)} folders -> {dest}: {merged.rows} rows, {totals['matches']} matches, "
          f"manifest sha256 {hashlib.sha256(blob).hexdigest()}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        p = argparse.ArgumentParser(prog="replay_26.py merge")
        p.add_argument("--out", required=True, help="output folder")
        p.add_argument("sources", nargs="+", help="--out folders of the shards")
        return merge(p.parse_args(sys.argv[2:]))

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output folder (must be empty)")
    p.add_argument("--data-dir", help="local copy of IL_Replay (folder with replays/*.parquet)")
    p.add_argument("--hf-cache", help="Hugging Face cache when downloading (default: ~/.cache)")
    p.add_argument("--chunks", help="chunks to read, e.g. 0-9,15 (default: all 52)")
    p.add_argument("--shard", type=int, default=0, help="index of this process (0..n-1)")
    p.add_argument("--num-shards", type=int, default=1, help="total number of processes")
    p.add_argument("--max-matches", type=int, default=10**9, help="stop after N replayed matches")
    args = p.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit("--shard must be between 0 and --num-shards - 1")
    replay(args)


if __name__ == "__main__":
    main()
