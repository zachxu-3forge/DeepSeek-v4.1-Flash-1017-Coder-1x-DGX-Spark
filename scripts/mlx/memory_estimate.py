#!/usr/bin/env python3
"""Estimate whether the target fits one 128 GB DGX Spark at a given context.

Weights are the checkpoint's resident size (Engram stays on SSD). The KV estimate
follows the engine's own layout: SWA layers keep `2 * window` rows, compressed
values/scales are stored once per KV source (reuse layers read the source), and
index keys once per index source. The 600K boot measured 99 GiB used / 22 GiB
available with this layout (2026-11-16), not the earlier per-layer bound.

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
DEFAULT_CONTEXTS = (4096, 16384, 32768, 65536, 131072, 262144, 600000, 1048576)


def kv_bytes_for(config: dict, context: int) -> int:
    text = config.get("text_config", config)
    layers = int(text.get("num_hidden_layers", 40))
    ratios = list(text.get("compress_ratios") or [])
    kv_sources = list(text.get("kv_source_layer_ids") or [])
    index_sources = list(text.get("index_source_layer_ids") or [])
    window = int(text.get("sliding_window", 128))
    index_dim = int(text.get("index_head_dim", 128))

    def ratio(layer: int) -> int:
        return int(ratios[layer]) if layer < len(ratios) else 0

    # SWA layers keep the last `window` tokens: values 576 B + scales 8 B a row.
    total = sum(2 * window * (576 + 8) for layer in range(layers) if ratio(layer) == 0)
    # Compressed values/scales once per KV source; reuse layers read that cache.
    for layer in kv_sources:
        r = max(ratio(layer), 1)
        total += (context // r + 1) * (576 + 8)
    # Index keys once per index source: 128 dims + 4 bytes.
    for layer in index_sources:
        r = max(ratio(layer), 1)
        total += (context // r + 1) * (index_dim + 4)
    # RoPE tables: (context + 2048) rows of 64 f32, window and compressed.
    total += (context + 2048) * 64 * 4 * 2
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--context", type=int, action="append", default=[])
    ap.add_argument("--weights-gib", type=float, default=88.6,
                    help="affine 83.37 GiB + dense 5.19 GiB (measured)")
    ap.add_argument("--overhead-gib", type=float, default=8.6,
                    help="buffers/workspace beyond weights/KV/rope (measured at 600K)")
    ap.add_argument("--reserve-gib", type=float, default=10.0, help="OS/driver/headroom")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    model = os.path.abspath(args.model_dir)
    with open(os.path.join(model, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    text = config.get("text_config", config)
    layers = int(text.get("num_hidden_layers", 40))
    contexts = args.context or list(DEFAULT_CONTEXTS)
    rows = []
    for ctx in contexts:
        kv = kv_bytes_for(config, ctx) / GIB
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
        "kv_layout": "swa layers + one compressed cache per KV source + index keys per index source + rope tables",
        "engram_on_ssd_gib": 114.4,
        "rows": rows,
        "note": "600K measured 99 GiB used / 22 GiB available on dgx2 (2026-11-16); "
                "remeasure after KV/prefill changes.",
    }
    text_out = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text_out + "\n")
    print(text_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
