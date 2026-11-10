#!/usr/bin/env python3
"""Estimate whether the target fits one 128 GB DGX Spark at a given context.

Weights are the checkpoint's resident size (Engram stays on SSD). KV uses the
engine's 584-byte record per layer per token, a conservative bound for the
compressed/SWA layers. Override the inputs when the live measurements arrive.

Example:
  python3 scripts/mlx/memory_estimate.py "$MODEL_DIR"
  python3 scripts/mlx/memory_estimate.py "$MODEL_DIR" --json out/memory.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

GIB = 1 << 30
DEFAULT_CONTEXTS = (4096, 16384, 32768, 65536, 131072, 262144, 1048576)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--context", type=int, action="append", default=[])
    ap.add_argument("--weights-gib", type=float, default=98.0)
    ap.add_argument("--overhead-gib", type=float, default=4.0, help="activations/workspace")
    ap.add_argument("--reserve-gib", type=float, default=10.0, help="OS/driver/headroom")
    ap.add_argument("--kv-bytes", type=int, default=584, help="per token per layer")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    model = os.path.abspath(args.model_dir)
    with open(os.path.join(model, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    text = config.get("text_config", config)
    layers = int(text.get("num_hidden_layers", 40))
    contexts = args.context or list(DEFAULT_CONTEXTS)
    kv_per_token = layers * args.kv_bytes
    rows = []
    for ctx in contexts:
        kv = ctx * kv_per_token / GIB
        total = args.weights_gib + args.overhead_gib + args.reserve_gib + kv
        headroom = 128.0 - total
        rows.append({
            "context": ctx,
            "kv_gib": round(kv, 2),
            "total_gib": round(total, 2),
            "headroom_gib": round(headroom, 2),
            "fits": headroom > 0,
        })

    result = {
        "model": model,
        "layers": layers,
        "weights_gib": args.weights_gib,
        "overhead_gib": args.overhead_gib,
        "reserve_gib": args.reserve_gib,
        "kv_bytes_per_token_per_layer": args.kv_bytes,
        "kv_bytes_per_token": kv_per_token,
        "engram_on_ssd_gib": 114.4,
        "rows": rows,
        "note": "KV is a conservative estimate; measure live before fixing CONTEXT.",
    }
    text_out = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text_out + "\n")
    print(text_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
