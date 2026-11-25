"""Export a checkpoint to ONNX, to play against it in the crforge app.

The file follows the format read by desktop/model/ModelPolicy.java (export_format 1):
inputs observations int32 (B, 7024) and masks bool (B, 2307), outputs first_logp (B, 7),
tile_logp (B, 4, 576) and value (B,). The metadata holds the schema and the vocabulary
(crforge.metadata), the 2.6 deck and its slots, and the level.

Usage:
  python models/export_onnx.py --model runs/ppo/ppo_last.pt --out runs/bot_26.onnx
Then, in sim/crforge:
  ./gradlew :desktop:run -Pcrforge.assetSource=cdn --args="--model /path/to/bot_26.onnx --model-sample"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.policy import load  # noqa: E402
from sim.cards import DECK_26, SLOTS_26  # noqa: E402


class Exported(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, observations, masks):
        return self.model(observations, masks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="bot 2.6")
    ap.add_argument("--level", type=int, default=16)
    ap.add_argument("--plain", action="store_true", help="deck without evolutions or heroes")
    ap.add_argument("--check", default=None, help="rows_*.npz file used to compare torch and ONNX")
    args = ap.parse_args()

    model, ck = load(args.model, "cpu")
    model.eval()
    wrapped = Exported(model)
    obs = torch.zeros(2, 7024, dtype=torch.int32)
    masks = torch.zeros(2, 2307, dtype=torch.bool)
    masks[:, 0] = True
    torch.onnx.export(
        wrapped, (obs, masks), args.out, input_names=["observations", "masks"],
        output_names=["first_logp", "tile_logp", "value"], opset_version=17, dynamo=False,
        dynamic_axes={"observations": {0: "batch"}, "masks": {0: "batch"}, "first_logp": {0: "batch"},
                      "tile_logp": {0: "batch"}, "value": {0: "batch"}},
    )
    m = onnx.load(args.out)
    slots = [0] * 8 if args.plain else SLOTS_26
    props = {
        "crforge.export_format": "1",
        "crforge.metadata": json.dumps(ck["metadata"]),
        "crforge.deck": ",".join(DECK_26),
        "crforge.deck_slots": ",".join(str(x) for x in slots),
        "crforge.level": str(args.level),
        "crforge.name": args.name,
    }
    del m.metadata_props[:]
    for k, v in props.items():
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, args.out)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")

    if args.check:
        import onnxruntime as ort

        d = np.load(args.check)
        o = d["obs"][:64].astype(np.int32)
        k = np.unpackbits(d["mask"][:64], axis=1)[:, :2307].astype(bool)
        sess = ort.InferenceSession(args.out)
        f_onnx, t_onnx, v_onnx = sess.run(None, {"observations": o, "masks": k})
        with torch.no_grad():
            f, t, v = model(torch.from_numpy(o), torch.from_numpy(k))
        open_first = f.numpy() > -1e8
        print("max diff first_logp", float(np.abs(f_onnx - f.numpy())[open_first].max()),
              "| tile_logp", float(np.abs(t_onnx - t.numpy())[t.numpy() > -1e8].max()),
              "| value", float(np.abs(v_onnx - v.numpy()).max()))


if __name__ == "__main__":
    main()
