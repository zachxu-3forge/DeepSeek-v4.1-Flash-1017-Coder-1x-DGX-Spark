#!/usr/bin/env python3
"""Pack the target's 4-bit affine Engram tables into the engine's row shards.

The engine reads `engram-l<layer>-r0of1.bin`:
  4096-byte header: magic "DSV41EN1", layer, lo=0, hi=rows, total=rows, row_bytes
  each row: `row_bytes` bytes, one of two formats:
    native (160): 32 U32 words of 4-bit codes, 8 BF16 scales, 8 BF16 biases (group 32)
    fp8    (264): 256 E4M3 codes, 8 UE8M0 scales (one per 32 values)

The target stores `U32 [rows,32]` 4-bit codes, BF16 `[rows,8]` scales, and BF16
`[rows,8]` biases. The default native mode copies those bytes row by row, with no
dequantization or requantization. `--format fp8` is the fallback: it dequantizes,
re-quantizes to E4M3 with a per-32 E8M0 scale, and streams the result. Both modes
need NumPy; neither needs a GPU.

Examples:
  python3 scripts/mlx/convert_engram.py "$MODEL_DIR" /engram-native --dry-run
  python3 scripts/mlx/convert_engram.py "$MODEL_DIR" /engram-native --verify
  python3 scripts/mlx/convert_engram.py "$MODEL_DIR" /engram-fp8 --format fp8 --verify
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np

MAGIC = 0x31344E4531565344  # "DSV41EN1"
HEADER_BYTES = 4096
VALUES = 256
SCALES = 8
GROUP = 32
NATIVE_ROW_BYTES = 160
FP8_ROW_BYTES = 264
ROW_BYTES = NATIVE_ROW_BYTES
ROW_OUT = ROW_BYTES


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        blob = f.read(n)
    return json.loads(blob), 8 + n


def read_rows(path: str, base: int, start: int, count: int, row_bytes: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(base + start * row_bytes)
        data = f.read(count * row_bytes)
    if len(data) != count * row_bytes:
        raise IOError(f"{path}: short read")
    return data


def bf16_to_f32(a: np.ndarray) -> np.ndarray:
    u = a.astype(np.uint32) << 16
    return u.view(np.float32)


def dequantize(words: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    # words [n,32] uint32 -> [n,256] 4-bit codes, low nibble first
    idx = np.arange(VALUES)
    q = (words[:, idx // 8] >> ((idx % 8) * 4)) & 0xF
    s = np.repeat(bf16_to_f32(scales), GROUP, axis=1)
    b = np.repeat(bf16_to_f32(biases), GROUP, axis=1)
    return q.astype(np.float32) * s + b


def native_chunk(wbytes: bytes, sbytes: bytes, bbytes: bytes) -> np.ndarray:
    """[n,160] native rows: 128 bytes of codes, then 8 BF16 scales and 8 BF16 biases."""
    n = len(wbytes) // (32 * 4)
    if len(sbytes) != n * 8 * 2 or len(bbytes) != n * 8 * 2:
        raise ValueError("native Engram chunk sizes differ")
    out = np.empty((n, NATIVE_ROW_BYTES), dtype=np.uint8)
    out[:, :128] = np.frombuffer(wbytes, dtype=np.uint8).reshape(n, 128)
    out[:, 128:144] = np.frombuffer(sbytes, dtype=np.uint8).reshape(n, 16)
    out[:, 144:160] = np.frombuffer(bbytes, dtype=np.uint8).reshape(n, 16)
    return out


def dequantize_native(rows: np.ndarray) -> np.ndarray:
    """[n,160] native rows -> [n,256] float32 values (the same code * scale + bias)."""
    n = rows.shape[0]
    words = np.frombuffer(rows[:, :128].tobytes(), dtype="<u4").reshape(n, 32)
    scales = np.frombuffer(rows[:, 128:144].tobytes(), dtype="<u2").reshape(n, 8)
    biases = np.frombuffer(rows[:, 144:160].tobytes(), dtype="<u2").reshape(n, 8)
    return dequantize(words, scales, biases)


def e4m3_bits(x: np.ndarray) -> np.ndarray:
    """Round float32 to E4M3FN (RNE), returning uint8 bit patterns."""
    sign = np.signbit(x).astype(np.uint8) << 7
    a = np.minimum(np.abs(x), 448.0)
    zero = a == 0.0
    m, exp = np.frexp(a)  # a = m * 2^exp, m in [0.5, 1)
    e = exp.astype(np.int32) - 1
    mant = np.rint((m * 2.0 - 1.0) * 8.0).astype(np.int32)
    carry = mant == 8
    e = e + carry.astype(np.int32)
    mant = np.where(carry, 0, mant)

    normal = e >= -6
    sub = np.rint(a / (2.0 ** -9)).astype(np.int32)
    sub_carry = (sub >= 8) & ~normal
    sub = np.where(sub_carry, 0, sub)
    e = np.where(sub_carry, -6, e)
    normal = normal | sub_carry

    exp_field = np.where(normal, e + 7, 0)
    mant_out = np.where(normal, mant, np.clip(sub, 0, 7))
    over = (exp_field > 15) | ((exp_field == 15) & (mant_out > 6))
    exp_field = np.where(over, 15, exp_field)
    mant_out = np.where(over, 6, mant_out)
    exp_field = np.clip(exp_field, 0, 15)
    bits = sign | (exp_field.astype(np.uint8) << 3) | mant_out.astype(np.uint8)
    return np.where(zero, 0, bits)


def e4m3_to_f32(bits: np.ndarray) -> np.ndarray:
    sign = np.where((bits >> 7) & 1, -1.0, 1.0)
    exp = ((bits >> 3) & 0xF).astype(np.int32)
    mant = (bits & 7).astype(np.float32) / 8.0
    normal = exp != 0
    e = np.where(normal, exp.astype(np.float32) - 7.0, -6.0)
    frac = np.where(normal, 1.0 + mant, mant)
    return sign * frac * np.exp2(e)


def quantize_row(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[n,256] float -> [n,256] E4M3 bytes, [n,8] E8M0 bytes."""
    n = vals.shape[0]
    g = vals.reshape(n, SCALES, GROUP)
    maxabs = np.max(np.abs(g), axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.ceil(np.log2(np.maximum(maxabs, 1e-30) / 448.0))
    e = np.where(maxabs > 0, e, 0.0)
    e = np.clip(e, -127, 127)
    scale = np.exp2(e)
    codes = e4m3_bits(g / scale[..., None]).reshape(n, VALUES)
    scale_bits = (e + 127).astype(np.uint8)
    return codes, scale_bits


def dequantize_fp8(codes: np.ndarray, scale_bits: np.ndarray) -> np.ndarray:
    n = codes.shape[0]
    s = np.repeat(np.exp2(scale_bits.astype(np.int32) - 127).astype(np.float32), GROUP, axis=1)
    return e4m3_to_f32(codes) * s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--limit-rows", type=int, default=0, help="convert only the first N rows")
    ap.add_argument("--chunk-rows", type=int, default=4096)
    ap.add_argument("--format", choices=("native", "fp8"), default="native", help="row format (default native)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    row_bytes = NATIVE_ROW_BYTES if args.format == "native" else FP8_ROW_BYTES
    model = os.path.abspath(args.model_dir)
    out = os.path.abspath(args.out_dir)
    config = load_json(os.path.join(model, "config.json"))
    text = config.get("text_config", config)
    layers = text.get("engram_layer_ids", [])
    rows_cfg = text.get("engram_num_embeddings", [])
    qmods = config.get("omlx_deepseek_v41", {}).get("quantized_modules", {})
    index = load_json(os.path.join(model, "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    headers: dict[str, tuple[dict, int]] = {}

    jobs = []
    total = 0
    for layer, want_rows in zip(layers, rows_cfg):
        base = f"language_model.layers.{layer}.engram.embed"
        spec = qmods.get(base)
        if spec is None:
            if base + ".weight" not in weight_map:
                print(f"skip layer {layer}: no Engram table", file=sys.stderr)
                continue
            spec = {"bits": 4, "group_size": 32, "mode": "affine"}
            print(f"layer {layer}: inferred 4-bit/group-32 table", file=sys.stderr)
        if spec.get("mode", "affine") != "affine" or spec.get("bits") != 4 or spec.get("group_size", 32) != 32:
            print(f"layer {layer}: expected affine 4-bit/group-32, got {spec}", file=sys.stderr)
            return 2
        wname = base + ".weight"
        shard = weight_map[wname]
        if shard not in headers:
            headers[shard] = read_header(os.path.join(model, shard))
        header, base_off = headers[shard]
        shape = header[wname]["shape"]
        if shape[0] != want_rows:
            print(f"layer {layer}: {shape[0]} rows, config says {want_rows}", file=sys.stderr)
            return 2
        rows = want_rows if args.limit_rows <= 0 else min(want_rows, args.limit_rows)
        jobs.append((layer, base, shard, base_off, rows))
        total += HEADER_BYTES + rows * row_bytes

    if not jobs:
        print("no Engram tables found", file=sys.stderr)
        return 2

    if args.dry_run:
        for layer, _base, _shard, _off, rows in jobs:
            print(f"layer {layer}: {rows} rows -> {HEADER_BYTES + rows * row_bytes} bytes ({row_bytes} B rows)")
        print(f"total: {total / (1 << 30):.2f} GiB")
        return 0

    probe = out
    while not os.path.isdir(probe):
        parent = os.path.dirname(probe) or "."
        if parent == probe:
            break
        probe = parent
    free = shutil.disk_usage(probe).free
    if free < total:
        print(f"not enough free space: need {total / (1 << 30):.2f} GiB, have {free / (1 << 30):.2f} GiB", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)

    for layer, base, shard, base_off, rows in jobs:
        path = os.path.join(out, f"engram-l{layer}-r0of1.bin")
        if os.path.exists(path) and not args.overwrite:
            print(f"exists: {path} (use --overwrite)", file=sys.stderr)
            return 2
        header = struct.pack("<6Q", MAGIC, layer, 0, rows, rows, row_bytes) + b"\0" * (HEADER_BYTES - 48)
        wname = base + ".weight"
        sname = base + ".scales"
        bname = base + ".biases"
        src = os.path.join(model, shard)
        done = 0
        tmp = path + ".part"
        with open(tmp, "wb") as dst:
            dst.write(header)
            while done < rows:
                n = min(args.chunk_rows, rows - done)
                wbytes = read_rows(src, base_off + header_offset(wname, headers[shard][0]), done, n, 32 * 4)
                sbytes = read_rows(src, base_off + header_offset(sname, headers[shard][0]), done, n, 8 * 2)
                bbytes = read_rows(src, base_off + header_offset(bname, headers[shard][0]), done, n, 8 * 2)
                words = np.frombuffer(wbytes, dtype="<u4").reshape(n, 32)
                scales = np.frombuffer(sbytes, dtype="<u2").reshape(n, 8)
                biases = np.frombuffer(bbytes, dtype="<u2").reshape(n, 8)
                vals = dequantize(words, scales, biases)
                if not np.isfinite(vals).all():
                    raise ValueError(f"layer {layer} rows {done}..{done+n}: non-finite values")
                if args.format == "native":
                    out_rows = native_chunk(wbytes, sbytes, bbytes)
                else:
                    codes, scale_bits = quantize_row(vals)
                    out_rows = np.empty((n, row_bytes), dtype=np.uint8)
                    out_rows[:, :VALUES] = codes
                    out_rows[:, VALUES:] = scale_bits
                dst.write(out_rows.tobytes())
                done += n
                print(f"layer {layer}: {done}/{rows} rows", file=sys.stderr)
        os.replace(tmp, path)
        if args.verify:
            max_err, sum_sq, count = verify_file(
                path, src, base_off, headers[shard][0], wname, sname, bname, rows, row_bytes, args.format
            )
            print(
                f"layer {layer} verify ({args.format}, {count} values): "
                f"max {max_err:.6g} rms {(sum_sq / count) ** 0.5:.6g}",
                file=sys.stderr,
            )
        print(f"wrote {path} ({rows} rows)")

    return 0


def verify_file(
    path: str,
    src: str,
    base_off: int,
    header: dict,
    wname: str,
    sname: str,
    bname: str,
    rows: int,
    row_bytes: int,
    fmt: str,
    sample: int = 64,
) -> tuple[float, float, int]:
    """Decode sampled rows from the written shard and compare with the source rows."""
    picks = sorted(set([0, rows - 1] + [i * rows // max(sample, 1) for i in range(max(sample, 1))]))
    max_err = 0.0
    sum_sq = 0.0
    count = 0
    for r in picks:
        if r < 0 or r >= rows:
            continue
        raw = read_rows(path, HEADER_BYTES, r, 1, row_bytes)
        wbytes = read_rows(src, base_off + header_offset(wname, header), r, 1, 32 * 4)
        sbytes = read_rows(src, base_off + header_offset(sname, header), r, 1, 8 * 2)
        bbytes = read_rows(src, base_off + header_offset(bname, header), r, 1, 8 * 2)
        words = np.frombuffer(wbytes, dtype="<u4").reshape(1, 32)
        scales = np.frombuffer(sbytes, dtype="<u2").reshape(1, 8)
        biases = np.frombuffer(bbytes, dtype="<u2").reshape(1, 8)
        want = dequantize(words, scales, biases)
        row = np.frombuffer(raw, dtype=np.uint8).reshape(1, row_bytes)
        if fmt == "native":
            got = dequantize_native(row)
        else:
            got = dequantize_fp8(row[:, :VALUES], row[:, VALUES:])
        err = np.abs(got - want)
        max_err = max(max_err, float(err.max()))
        sum_sq += float(np.sum(err.astype(np.float64) ** 2))
        count += err.size
    if count == 0:
        raise ValueError(f"{path}: no rows sampled")
    return max_err, sum_sq, count


def header_offset(name: str, header: dict) -> int:
    return header[name]["data_offsets"][0]


if __name__ == "__main__":
    raise SystemExit(main())
