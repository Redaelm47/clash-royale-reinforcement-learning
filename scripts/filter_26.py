"""Find 2.6 Hog Cycle matches in IL_Replay replays/*.parquet.
Usage: python3 -I filter_26.py OUT_DIR SHARD.parquet [SHARD.parquet ...]
A deck is 2.6 if, after stripping the variant suffix (-ev1, -ev2, ..., -hero)
from each deck card_key, it is exactly the 8 base keys below."""
import sys, os, json, re
from collections import Counter
import pyarrow as pa, pyarrow.parquet as pq

BASE_26 = {"hog-rider", "musketeer", "cannon", "ice-golem",
           "ice-spirit", "skeletons", "fireball", "the-log"}
SUFFIX = re.compile(r"-(ev\d+|hero)$")
def base(k): return SUFFIX.sub("", k)
def is_26(deck): return len(deck) == 8 and {base(c["card_key"]) for c in deck} == BASE_26

out_dir, shards = sys.argv[1], sys.argv[2:]
os.makedirs(out_dir, exist_ok=True)
n = 0; any26 = 0; mirror = 0; nplayers26 = 0
by_side = Counter(); by_bt = Counter(); variants = Counter(); levels = Counter(); res = Counter()
variant_keys = Counter(); slot_order = Counter()
keep = []
for p in shards:
    for b in pq.ParquetFile(p).iter_batches(batch_size=2000):
        for row in b.to_pylist():
            n += 1
            bt = json.loads(row["payload_json"])["battle"]
            for side in ("team", "opponent"):
                for pl in bt[side]["players"]:
                    for c in pl["deck"]:
                        if base(c["card_key"]) in BASE_26: variant_keys[c["card_key"]] += 1
            s26 = [side for side in ("team", "opponent") if is_26(bt[side]["players"][0]["deck"])]
            if not s26: continue
            any26 += 1; mirror += len(s26) == 2; nplayers26 += len(s26)
            by_bt[(row["battle_type"], row["game_mode"])] += 1
            for side in s26:
                by_side[side] += 1
                deck = bt[side]["players"][0]["deck"]
                variants[tuple(sorted(c["card_key"] for c in deck if c["card_key"] != base(c["card_key"])))] += 1
                levels[tuple(sorted({c["level"] for c in deck}))] += 1
                slot_order[tuple(c["card_key"] for c in deck)[:3]] += 1
                # result from the 2.6 player's point of view
                r = row["result"] if side == "team" else {"victory": "defeat", "defeat": "victory"}.get(row["result"], row["result"])
                res[r] += 1
            row["hog26_sides"] = ",".join(s26)
            keep.append(row)
pq.write_table(pa.Table.from_pylist(keep), os.path.join(out_dir, "hog26_replays.parquet"))
summary = dict(matches_scanned=n, matches_with_a_26_deck=any26, mirror_matches=mirror,
               player_decks_26=nplayers26, by_side=dict(by_side),
               by_battle_type_mode={f"{a}/{b}": v for (a, b), v in by_bt.most_common()},
               results_from_26_player=dict(res),
               evo_hero_variants_in_26_decks={("+".join(k) or "none"): v for k, v in variants.most_common()},
               level_sets_in_26_decks={str(k): v for k, v in levels.most_common(10)},
               first3_slots_26={" | ".join(k): v for k, v in slot_order.most_common(8)},
               all_deck_keys_for_the_8_cards=dict(variant_keys.most_common()))
json.dump(summary, open(os.path.join(out_dir, "hog26_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
