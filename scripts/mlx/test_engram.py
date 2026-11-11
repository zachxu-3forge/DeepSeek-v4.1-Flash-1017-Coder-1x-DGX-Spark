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
import struct
import sys

import reference_layer as rl

MAGIC = 0x31344E4531565344  # "DSV41EN1"
HEADER_BYTES = 4096
NATIVE_ROW_BYTES = 160
FP8_ROW_BYTES = 264
FP8_VALUES = 256
FP8_SCALES = 8
FP8_GROUP = 32


def check_shard(shard_dir: str, layer: int, rows: int, values: list[float], words: bytes, scales: bytes, biases: bytes) -> None:
    path = os.path.join(shard_dir, f"engram-l{layer}-r0of1.bin")
    with open(path, "rb") as f:
        head = f.read(48)
        magic, got_layer, lo, hi, total, row_bytes = struct.unpack("<6Q", head)
        if magic != MAGIC or got_layer != layer or lo != 0 or hi != rows or total != rows:
            raise ValueError(f"{path}: bad header")
        if row_bytes not in (NATIVE_ROW_BYTES, FP8_ROW_BYTES):
            raise ValueError(f"{path}: unsupported row size {row_bytes}")
        f.seek(HEADER_BYTES)
        row = f.read(row_bytes)
    if len(row) != row_bytes:
        raise ValueError(f"{path}: short row")
    if row_bytes == NATIVE_ROW_BYTES:
        if row[0:128] != words or row[128:144] != scales or row[144:160] != biases:
            src = words + scales + biases
            at = next((i for i, (x, y) in enumerate(zip(row, src)) if x != y), min(len(row), len(src)))
            raise ValueError(
                f"{path}: row 0 bytes differ at {at} "
                f"(shard {row[max(0, at - 4):at + 4].hex()} vs source {src[max(0, at - 4):at + 4].hex()})"
            )
        got = rl.decode(row[0:128], row[128:144], row[144:160], 4, 256, 32)
        tol = 1e-6
    else:
        got = rl.decode(row[:FP8_VALUES], row[FP8_VALUES:], None, 8, FP8_VALUES, FP8_GROUP)
        tol = 0.01 * max(1.0, max(abs(v) for v in values))
    diff = max(abs(a - b) for a, b in zip(got, values))
    if diff > tol:
        raise ValueError(f"{path}: row 0 differs from the release by {diff} (tol {tol})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--out")
    ap.add_argument("--shard-dir", help="also check row 0 of each packed shard against the release")
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
            if name + ".weight" not in weight_map:
                print(f"missing {name}: no weight tensor", file=sys.stderr)
                return 2
            spec = {"bits": 4, "group_size": 32, "mode": "affine"}
            print(f"{name}: inferred 4-bit/group-32 table", file=sys.stderr)
        if spec.get("mode", "affine") != "affine" or spec.get("bits") != 4 or spec.get("group_size", 32) != 32:
            print(f"{name}: expected affine 4-bit/group-32, got {spec}", file=sys.stderr)
            return 2
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
        if args.shard_dir:
            check_shard(os.path.abspath(args.shard_dir), layer, want_rows, values, words, scales, biases)
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

    if not results:
        print("no Engram tables found", file=sys.stderr)
        return 2

    text_out = json.dumps(results, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text_out + "\n")
    print(text_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
