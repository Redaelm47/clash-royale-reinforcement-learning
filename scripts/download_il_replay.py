"""Downloads the 52 raw IL_Replay files (about 0.83 GB, 260,000 games) and checks their SHA-256.

Source: https://huggingface.co/datasets/VanguardX101/IL_Replay (replays/ folder).
A file that is already there and correct is not downloaded again.

Usage:
  python scripts/download_il_replay.py data/replays/il_replay_all
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/VanguardX101/IL_Replay/resolve/main/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/replays/il_replay_all")
    out.mkdir(parents=True, exist_ok=True)
    sums = urllib.request.urlopen(BASE + "SHA256SUMS", timeout=60).read().decode()
    wanted = [(h, p) for h, p in (line.split() for line in sums.splitlines() if line.strip()) if p.startswith("replays/")]
    print(f"{len(wanted)} files expected in {out}", flush=True)
    bad = []
    for h, p in wanted:
        dest = out / Path(p).name
        if dest.exists() and sha256(dest) == h:
            continue
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(BASE + p, tmp)
        if sha256(tmp) != h:
            bad.append(p)
            tmp.unlink()
            print(f"  {p}: wrong SHA-256 sum, skipped", flush=True)
            continue
        tmp.replace(dest)
        print(f"  {p} ok", flush=True)
    n = sum(1 for _ in out.glob("part-*.parquet"))
    print(f"done: {n} correct files, {len(bad)} failed", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
