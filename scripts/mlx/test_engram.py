#!/usr/bin/env python3
"""Golden values for the target's 4-bit Engram tables.

Reads row 0 of each `engram.embed` table, checks the row count against
`engram_num_embeddings`, decodes 4-bit/group-32 affine, and writes deterministic dots.
The future affine Engram reader/converter must reproduce these.

Example:
  python3 scripts/mlx/test_engram.py "$MODEL_DIR" --out out/engram.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import reference_layer as rl


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    model = os.path.abspath(args.model_dir)
    config = rl.load_json(os.path.join(model, "config.json"))
    text = config.get("text_config", config)
    layers = text.get("engram_layer_ids", [])
    rows = text.get("engram_num_embeddings", [])
    if len(layers) != len(rows):
        print("engram_layer_ids and engram_num_embeddings differ in length", file=sys.stderr)
        return 2
    qmods = config.get("omlx_deepseek_v41", {}).get("quantized_modules", {})
    index = rl.load_json(os.path.join(model, "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    headers: dict[str, tuple[dict, int]] = {}

    results = []
    for layer, want_rows in zip(layers, rows):
        name = f"language_model.layers.{layer}.engram.embed"
        spec = qmods.get(name)
        if spec is None:
            print(f"skip {name}: not in the quantization map", file=sys.stderr)
            continue
        weight_name = name + ".weight"
        shard = weight_map[weight_name]
        if shard not in headers:
            headers[shard] = rl.read_header(os.path.join(model, shard))
        header, base = headers[shard]
        shape = header[weight_name]["shape"]
        if shape[0] != want_rows:
            print(f"{name}: {shape[0]} rows, config says {want_rows}", file=sys.stderr)
            return 1
        bits = spec["bits"]
        group = spec.get("group_size", 32)
        k = shape[-1] * 32 // bits
        words = rl.read_row(model, shard, header, base, weight_name, 0)
        scales = rl.read_row(model, shard, header, base, name + ".scales", 0)
        biases = rl.read_row(model, shard, header, base, name + ".biases", 0)
        values = rl.decode(words, scales, biases, bits, k, group)
        x1 = [1.0] * k
        x2 = [((i % 11) - 5) * 0.5 for i in range(k)]
        results.append({
            "module": name,
            "rows": shape[0],
            "bits": bits,
            "group": group,
            "k": k,
            "dot_ones": rl.dot(values, x1),
            "dot_alt": rl.dot(values, x2),
            "first": values[:8],
        })

    text_out = json.dumps(results, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text_out + "\n")
    print(text_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
