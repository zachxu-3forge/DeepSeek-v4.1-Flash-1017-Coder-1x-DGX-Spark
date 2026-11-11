#!/usr/bin/env python3
"""Golden CPU values for selected Stage 1 modules.

Reads one row per module straight from the safetensors shards, decodes the MLX bit
stream independently of the Zig reference, and writes deterministic dots. The
future GPU one-layer harness must reproduce these numbers.

Examples:
  python3 scripts/mlx/reference_layer.py "$MODEL_DIR" --out out/reference_layer.json
  python3 scripts/mlx/reference_layer.py "$MODEL_DIR" --module language_model.layers.0.ffn.experts.w1 --expert 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

PREFIXES = ("model.language_model.", "language_model.", "text_model.", "model.")

DEFAULT_MODULES = [
    "language_model.layers.0.attn.wq_a",
    "language_model.layers.0.ffn.shared_experts.w1",
    "language_model.layers.0.ffn.experts.w1",
    "language_model.mtp.0.attn.wq_a",
    "language_model.mtp.0.ffn.experts.w1",
]


def canonical(key: str) -> str:
    k = key
    if k.endswith(".weight"):
        k = k[: -len(".weight")]
    for p in PREFIXES:
        if k.startswith(p):
            k = k[len(p):]
    return k


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dtype_size(dtype: str) -> int:
    return {"U32": 4, "BF16": 2, "U8": 1}[dtype]


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        blob = f.read(n)
    return json.loads(blob), 8 + n


def read_row(model_dir: str, shard: str, header: dict, base: int, name: str, flat_row: int) -> bytes:
    info = header[name]
    shape = info["shape"]
    row = shape[-1] * dtype_size(info["dtype"])
    begin = info["data_offsets"][0] + flat_row * row
    with open(os.path.join(model_dir, shard), "rb") as f:
        f.seek(base + begin)
        data = f.read(row)
    if len(data) != row:
        raise ValueError(f"{name}: short row read")
    return data


def code_at(data: bytes, bit: int, bits: int) -> int:
    at = bit // 8
    v = 0
    for i in range(3):
        if at + i < len(data):
            v |= data[at + i] << (8 * i)
    return (v >> (bit % 8)) & ((1 << bits) - 1)


def bf16(data: bytes, i: int) -> float:
    h = struct.unpack_from("<H", data, 2 * i)[0]
    return struct.unpack("<f", struct.pack("<I", h << 16))[0]


def e8m0(byte: int) -> float:
    if byte == 0:
        return 2.0 ** -127
    if byte == 0xFF:
        return float("nan")
    return 2.0 ** (byte - 127)


def e2m1(code: int) -> float:
    """MXFP4 E2M1: sign bit 3; magnitude {0, .5, 1, 1.5, 2, 3, 4, 6}."""
    mag = code & 0x7
    value = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[mag]
    return -value if code & 0x8 else value


def e4m3(bits: int) -> float:
    """MXFP8 E4M3FN: E4M3 with 448 max; 0x7F / 0xFF are NaN."""
    sign = -1.0 if bits & 0x80 else 1.0
    exp = (bits >> 3) & 0xF
    mant = bits & 0x7
    if exp == 0xF and mant == 0x7:
        return float("nan")
    if exp == 0:
        return sign * (mant / 8.0) * 2.0 ** -6
    return sign * (1.0 + mant / 8.0) * 2.0 ** (exp - 7)


def decode(words: bytes, scales: bytes, biases: bytes | None, bits: int, k: int, group: int) -> list[float]:
    groups = k // group
    out = []
    for i in range(k):
        q = code_at(words, i * bits, bits)
        g = i // group
        if biases is None:
            # MXFP8/MXFP4 codes are small floats, not unsigned integers.
            if bits == 4:
                value = e2m1(q)
            elif bits == 8:
                value = e4m3(q)
            else:
                raise ValueError(f"mixed decode: unsupported {bits}-bit code")
            out.append(value * e8m0(scales[g]))
        else:
            s = bf16(scales, g)
            b = bf16(biases, g)
            out.append(q * s + b)
    return out


def dot(values: list[float], x: list[float]) -> float:
    return math.fsum(v * xi for v, xi in zip(values, x))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--module", action="append", default=[], help="module name (repeatable)")
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--out", help="write JSON here")
    args = ap.parse_args(argv)

    model = os.path.abspath(args.model_dir)
    config = load_json(os.path.join(model, "config.json"))
    qmods = config.get("omlx_deepseek_v41", {}).get("quantized_modules", {})
    index = load_json(os.path.join(model, "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    headers: dict[str, tuple[dict, int]] = {}

    modules = args.module or DEFAULT_MODULES
    results = []
    for name in modules:
        key = canonical(name)
        spec = qmods.get(name) or qmods.get(key)
        if spec is None:
            print(f"skip {name}: not in the quantization map", file=sys.stderr)
            continue
        weight_name = name + ".weight"
        if weight_name not in weight_map:
            print(f"skip {name}: no weight tensor", file=sys.stderr)
            continue
        shard = weight_map[weight_name]
        if shard not in headers:
            headers[shard] = read_header(os.path.join(model, shard))
        header, base = headers[shard]
        info = header[weight_name]
        shape = info["shape"]
        bits = spec["bits"]
        mode = spec.get("mode", "affine")
        n = shape[-2]
        packed = shape[-1]
        k = packed * 32 // bits
        group = spec.get("group_size", 64) if mode == "affine" else 32
        if k % group != 0:
            raise ValueError(f"{name}: K={k} not divisible by group {group}")
        groups = k // group
        flat = (args.expert * n + args.row) if len(shape) == 3 else args.row
        words = read_row(model, shard, header, base, weight_name, flat)
        scales_name = name + ".scales"
        scales = read_row(model, shard, header, base, scales_name, flat)
        biases = None
        if mode == "affine":
            biases = read_row(model, shard, header, base, name + ".biases", flat)
        values = decode(words, scales, biases, bits, k, group)
        x1 = [1.0] * k
        x2 = [((i % 11) - 5) * 0.5 for i in range(k)]
        results.append({
            "module": name,
            "canonical": key,
            "mode": mode,
            "bits": bits,
            "group": group,
            "n": n,
            "k": k,
            "expert": args.expert if len(shape) == 3 else None,
            "row": args.row,
            "dot_ones": dot(values, x1),
            "dot_alt": dot(values, x2),
            "first": values[:8],
        })

    text = json.dumps(results, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
